import os
import json
import math
import time
from copy import copy
from enum import Enum
from pprint import pprint
from typing import Any, Annotated
from typing import Iterable

import instructor
import networkx as nx
import numpy as np
import pandas as pd
import requests
import tiktoken
from bs4 import BeautifulSoup
from lxml import html
from lxml.etree import XPath, XPathSyntaxError
from matplotlib import pyplot as plt
from openai import OpenAI
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    ConfigDict,
    model_validator,
    FieldValidationInfo,
    AfterValidator
)

print("Setting up!! :)")


class KitapException(Exception):
    pass


kitap_host = os.environ["KITAP_HOST"]

client = instructor.patch(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
vision_client = instructor.patch(OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
                                 mode=instructor.Mode.MD_JSON)

supported_actions_metadata = pd.read_excel("./WebActions.xlsx", usecols=["Actions(Web)", "Description"])
supported_actions_metadata.columns = ["Action", "Description"]

preprocess_action = lambda s: s.title().strip().replace(" ", "")

preprocessed_actions = supported_actions_metadata["Action"].apply(preprocess_action)
original_action_mapping = {k: v for k, v in zip(preprocessed_actions, supported_actions_metadata["Action"])}

assert len(supported_actions_metadata["Action"]) == len(original_action_mapping)

supported_actions_metadata["Action"] = preprocessed_actions
supported_actions_metadata.head()


def _must_be_valid_xpath_expression(ui_element: str | None):
    if ui_element is None:
        return

    def is_valid_xpath(expression):
        try:
            XPath(expression)
            return True
        except XPathSyntaxError:
            return False

    if not is_valid_xpath(ui_element):
        raise ValueError("UI Element must contain a valid XPath expression")

    return ui_element


Action = Enum("Action", {k: str(v) for k, v in zip(supported_actions_metadata.Action.tolist(),
                                                   range(len(supported_actions_metadata.Action.tolist())))})


class KitapResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    url: HttpUrl
    screenshot_s3_path: str
    page_source: BeautifulSoup
    tree: Any = None

    def model_post_init(self, *args, **kwargs):
        self.tree = html.fromstring(str(self.page_source)) if str(self.page_source) else None

    @property
    def body(self) -> str:
        page_source = copy(self.page_source)
        [x.extract() for x in page_source.findAll('script')]
        return str(page_source.body)


class TestStep(BaseModel):
    # chain_of_thought: str = Field(description="Think step-by-step to correctly create a test step")
    name: str
    ui_element: Annotated[str | None, AfterValidator(_must_be_valid_xpath_expression)]
    action: Action = Field(description=f"Action information:\n\n{supported_actions_metadata.to_json(orient='records')}")
    data: str | None = Field(
        description="Input to any UI elements that require it. For example, search fields might require the search keywords, etc. On the contrary, button clicks do not require any data, thus in that case this should be None.")
    is_terminal_step: bool = Field(
        description="Whether there needs to be a next step or if the testing scenario has terminated")

    def __hash__(self):
        return hash(self.name)

    @property
    def kitap_action(self) -> str:
        return original_action_mapping[self.action.name]

    @model_validator(mode="after")
    def validate_ui_element(self, info: FieldValidationInfo) -> "TestStep":

        if not info.context:
            # print("--> Not validating!! :(")
            return self

        kitap_response: KitapResponse = info.context.get("kitap_response")
        if not self.ui_element or kitap_response is None or not str(kitap_response.page_source):
            print("--> Still not validating!! :(")
            return self

        assert kitap_response.tree.xpath(self.ui_element), "The predicted XPath not found in the page source"


with open("./sample-kitap-response.json") as f:
    sample_kitap_response_dict = json.load(f)

sample_kitap_response = KitapResponse(
    page_source=BeautifulSoup(sample_kitap_response_dict.pop("page_source"), "html.parser"),
    **sample_kitap_response_dict
)

example_df = pd.read_excel("./TC_22_Standard Ship Order_Steps.xlsx", usecols=["Name", "UI Element", "Action", "Data"])
example_df.Action = example_df.Action.apply(preprocess_action)
example_df.Data = example_df.Data.apply(lambda x: str(x) if not pd.isna(x) else x)
example_df = example_df.replace({np.nan: None})
example_df.head()

example_steps = [
    TestStep(
        name=row["Name"],
        ui_element=row["UI Element"],
        action=Action[row["Action"]],
        data=row["Data"],
        is_terminal_step=idx == len(example_df) - 1
    ) for idx, row in example_df.iterrows()]

system_prompt = [{
    "role": "system",
    "content": f"You are a world-class webpage traversal agent. Given a start URL, "
               f"you can mock all the actions and inputs required to thoroughly traverse "
               f"the entire web application. Your abilities include state-of-the-art mock "
               f"data and step generation which helps you perform actions which in turn "
               f"affect the state of the web-application. Given the current state of the web-app, you should output a wide range of test-steps that will affect the web-app in different ways. For example, if the current state was a login page, you will predict a list of test steps, out of which one might be correct login credentials, second might be incorrect login credentials, third might be sign-up, fourth might be forgot password, etc. Note that this is a recursive process. Although you operate according to the REST protocol, you will receive the history of all the test steps you have already predicted in order for you to have reached the current page (unless ofcourse the current page is the start URL). Always ensure coverage i.e., ensure that a wide range of the web-app's functionality is covered by your steps. All of the test-steps predict will be constructed as a directional graph or a tree. For example, the root will be the start URL, its children will be each of the different test-steps you decide to take on that start URL, and the children of each child will be all of the test steps you decide to take on each of the child page, and so on. Therefore, you must be careful not to run into any infinite loops. When you think the series of test-steps you have predicted along a single path has reached its logical end, stop gracefully, and continue with other branches.\n\nImportant note: The list of test steps you predict for the current state have to be significantly different from each other and be meaningful. For example, a scrolling action might not be meaningful but clicking on different buttons might be. For example, when traversing through a single path of the graph, your test steps might look like this: " + "\n".join(
        [f"\t{idx + 1}. {repr(example_step)}" for idx, example_step in enumerate(
            example_steps)]) + "\n\nNOTE: the above example is only along root to leaf of the directional graph. They follow a logical order because of the traversal order. Your outputs in one message will not look like this since you are to output test steps of varying logical flows. They will only make sense during traversal, not necessarily during your generation process. You have to ensure that all of the test steps you predict follow the logic of the list of test steps provided to you (which you have predicted earlier), but should not follow each other. For each iteration, do not predict more than 5 steps. If you have reached enough depth, where the order of test steps stops making logical sense, stop by setting `is_terminal_step` to `True`"
}]


def create_session() -> str:
    response = requests.post(f"{kitap_host}/v1/start-session?browser=chrome")
    response.raise_for_status()
    return response.json()["sessionId"]


def end_session(session_id: str) -> str:
    response = requests.post(f"{kitap_host}/v1/close-session?sessionId={session_id}")
    response.raise_for_status()
    return response.text


def execute_step(test_step: TestStep, session_id: str) -> KitapResponse:
    payload = {
        "sessionId": session_id,
        "name": test_step.name,
        "action": test_step.kitap_action,
        "locator": test_step.ui_element,
        "data": test_step.data,
        "description": "",
        "screenshot": True
    }

    headers = {
        'Content-Type': 'application/json',
        'Accept': '*/*'
    }

    st_time = time.perf_counter()
    print("Waiting for KiTAP to respond....")
    response = requests.post(f"{kitap_host}/v1/execute-step", headers=headers, json=payload)
    print(f"KiTAP responded in {time.perf_counter() - st_time:.2f} seconds.")

    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        try:
            response_body = response.json()
            raise KitapException(response_body["remarks"])
        except (json.JSONDecodeError, KeyError):
            try:
                raise KitapException(response_body["errorMessage"])
            except KeyError:
                # print(response.text)
                raise KitapException(response.text)

    body = response.json()

    raw_html = body["content"]
    soup = BeautifulSoup(raw_html, "html.parser")

    # print(f'{body["currentUrl"]=}')

    return KitapResponse(
        url=body["currentUrl"],
        screenshot_s3_path=body["screenShotPath"],
        page_source=soup
    )


def execute_steps(test_steps: Iterable[TestStep]) -> KitapResponse:
    session_id = create_session()

    try:
        for test_step in test_steps:
            response = execute_step(test_step, session_id)

        return response
    finally:
        end_session(session_id)


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def generate_test_steps(graph: nx.Graph, last_test_step, path_test_steps, kitap_response: KitapResponse, url,
                        session_id=None):
    messages = (system_prompt
                + [{"role": "user", "content": url}]
                + [{"role": "assistant", "content": repr(test_step)} for test_step in path_test_steps]
                + [{"role": "system",
                    "content": f"Your suggested action has been executed. Here is the updated information:\n\nURL: {kitap_response.url}\nPage source (first 5000 chars):\n\n{kitap_response.body[:5000]}"}]
                )

    # pprint(f"###\n\n{messages}\n\n###")

    test_steps: Iterable[TestStep] = client.chat.completions.create(
        messages=messages,
        response_model=Iterable[TestStep],
        model="gpt-4",
        max_tokens=1000,
        temperature=0,
        max_retries=3
    )

    print(f"test steps: {[test_step.name for test_step in test_steps]}")

    for test_step in test_steps:
        # print(f"Executing {test_step}... ")
        try:
            # kitap_response = execute_step(test_step, session_id)
            kitap_response = execute_steps(path_test_steps + [test_step])
        except KitapException as e:
            try:
                test_step: TestStep = client.chat.completions.create(
                    messages=(messages
                              + [{"role": "assistant", "content": repr(test_step)}]
                              + [{"role": "system",
                                  "content": f"There was an error when trying to execute your last test step.\n\nError: {e}\n\nConsider this error and try to rephrase your last test step, or else change your approach to the problem."}]),
                    response_model=TestStep,
                    model="gpt-4",
                    max_tokens=1000,
                    temperature=0,
                    max_retries=3
                )
            except Exception as e:
                # print(e, "Stopping this test step...")
                continue

        # print("Done!")

        graph.add_edges_from([(last_test_step, test_step)])

        nx.draw_networkx(
            graph,
            font_size=6,
            pos=nx.spring_layout(graph, k=5 / math.sqrt(graph.order()), seed=42),
            labels={node: f"{node.name, node.action.name}" for node in graph.nodes if isinstance(node, TestStep)},
        )
        plt.show()

        if test_step.is_terminal_step:
            continue

        generate_test_steps(
            graph=graph,
            last_test_step=test_step,
            path_test_steps=list(nx.all_simple_paths(graph, "root", test_step))[0][1:],
            kitap_response=kitap_response,
            url=url,
            session_id=None
        )


def construct_graph(url):
    graph = nx.DiGraph()

    start_test_steps: list[TestStep] = [
        TestStep(
            name="Open Browser",
            ui_element=None,
            action=Action("28"),
            data="Chrome",
            is_terminal_step=False
        ),
        TestStep(
            name="Navigate",
            ui_element=None,
            action=Action("27"),
            data=url,
            is_terminal_step=False
        )
    ]

    session_id = create_session()

    kitap_responses: list[KitapResponse] = [KitapResponse(
        url="https://www.google.com/",
        screenshot_s3_path="",
        page_source=BeautifulSoup("", "html.parser")
    ), execute_step(start_test_steps[-1], session_id=session_id)]

    end_session(session_id)

    graph.add_edges_from([("root", start_test_steps[-2]), (start_test_steps[-2], start_test_steps[-1])])

    generate_test_steps(
        graph=graph,
        last_test_step=start_test_steps[-1],
        path_test_steps=list(nx.all_simple_paths(graph, "root", start_test_steps[-1]))[0][1:],
        kitap_response=kitap_responses[-1],
        url=url,
        session_id=session_id
    )

if __name__ == "__main__":
    graph = construct_graph("https://www.dentomart.com")
    
    nx.draw_networkx(
        graph,
        font_size=6,
        pos=nx.spring_layout(graph, k=5 / math.sqrt(graph.order()), seed=42),
        labels={node: f"{node.name, node.action.name}" for node in graph.nodes if isinstance(node, TestStep)},
        # edgelist=[edge for edge in graph.edges if edge[1].name != "Open Browser"]
    )
    
    [edge for edge in graph.edges if edge[1].name != "Open Browser"]
    
    # draw start url
    nx.draw_networkx(
        graph,
        with_labels=False,
        font_size=8,
        node_color="#008000",
        nodelist=[node for node in graph.nodes if url == node],
        edgelist=[edge for edge in graph.edges if url == edge[0]],
        arrows=True,
        pos=nx.spring_layout(graph, k=5 / math.sqrt(graph.order()), seed=42)
    )
    
    # draw others
    nx.draw_networkx(
        graph,
        with_labels=False,
        font_size=8,
        node_color="#0000FF",
        # nodelist=[],
        nodelist=[node for node in graph.nodes if url != node],
        # edgelist=[],
        arrows=True,
        pos=nx.spring_layout(graph, k=5 / math.sqrt(graph.order()), seed=42)
    )
    plt.show()
    
    end_session(session_id=session_id)


def downstream_predict() -> None:
    functional_llm_system_prompt = """You are a world-class functionality inferring agent. Given the graph representation of a web-application, you can predict all the functionalities supported by the application. Do not suggest generic functionalities. Only predict those that are specific to the user's web app. Carefully consider the graph's nodes and edges and come up with a list of BDD flows. Your flows must be in plain english, in the tone of an expert Business Analyst. For example, one flow for Amazon might be: Go to Amazon.com, navigate to the refunds page and claim refund"""

    class WebpageDetails(BaseModel):
        bdd_flow: list[str]

    response: WebpageDetails = client.chat.completions.create(
        messages=[{"role": "system", "content": functional_llm_system_prompt}]
                 + [{"role": "user",
                     "content": f"Given the below graph representation of my web-app, correctly predict all the functionalities it supports.\n\nNodes:\n {graph.nodes}\n\nEdges:\n{graph.edges}"}],
        model="gpt-4",
        response_model=WebpageDetails
    )

    pprint(response.bdd_flow)


if __name__ == "__main__":
    downstream_predict()
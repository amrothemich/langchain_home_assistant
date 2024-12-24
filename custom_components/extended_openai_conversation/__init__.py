"""The OpenAI Conversation integration using LangChain."""
from __future__ import annotations

import json
import logging
from typing import Literal, Any

from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage, FunctionMessage
from langchain.callbacks import AsyncCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.tools import StructuredTool
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory
from openai import AsyncOpenAI
from openai._exceptions import AuthenticationError, OpenAIError

import yaml

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import async_should_expose
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import ATTR_NAME, CONF_API_KEY, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import (
    ConfigEntryNotReady,
    HomeAssistantError,
    TemplateError,
)
from homeassistant.helpers import (
    config_validation as cv,
    entity_registry as er,
    intent,
    template,
)
from homeassistant.helpers.httpx_client import get_async_client
from homeassistant.helpers.typing import ConfigType
from homeassistant.util import ulid

from .const import (
    CONF_API_VERSION,
    CONF_ATTACH_USERNAME,
    CONF_BASE_URL,
    CONF_CHAT_MODEL,
    CONF_CONTEXT_THRESHOLD,
    CONF_CONTEXT_TRUNCATE_STRATEGY,
    CONF_FUNCTIONS,
    CONF_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    CONF_MAX_TOKENS,
    CONF_ORGANIZATION,
    CONF_PROMPT,
    CONF_SKIP_AUTHENTICATION,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    CONF_USE_TOOLS,
    DEFAULT_ATTACH_USERNAME,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CONF_FUNCTIONS,
    DEFAULT_CONTEXT_THRESHOLD,
    DEFAULT_CONTEXT_TRUNCATE_STRATEGY,
    DEFAULT_MAX_FUNCTION_CALLS_PER_CONVERSATION,
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROMPT,
    DEFAULT_SKIP_AUTHENTICATION,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_USE_TOOLS,
    DOMAIN,
    EVENT_CONVERSATION_FINISHED,
)
from .exceptions import (
    FunctionLoadFailed,
    FunctionNotFound,
    InvalidFunction,
    ParseArgumentsFailed,
    TokenLengthExceededError,
)
from .helpers import (
    get_function_executor,
    is_azure,
    validate_authentication,
)
from .services import async_setup_services

_LOGGER = logging.getLogger(__name__)

CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# hass.data key for agent.
DATA_AGENT = "agent"

async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up OpenAI Conversation."""
    await async_setup_services(hass, config)
    return True

async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up OpenAI Conversation from a config entry."""
    try:
        await validate_authentication(
            hass=hass,
            api_key=entry.data[CONF_API_KEY],
            base_url=entry.data.get(CONF_BASE_URL),
            api_version=entry.data.get(CONF_API_VERSION),
            organization=entry.data.get(CONF_ORGANIZATION),
            skip_authentication=entry.data.get(
                CONF_SKIP_AUTHENTICATION, DEFAULT_SKIP_AUTHENTICATION
            ),
        )
    except AuthenticationError as err:
        _LOGGER.error("Invalid API key: %s", err)
        return False
    except OpenAIError as err:
        raise ConfigEntryNotReady(err) from err

    agent = OpenAIAgent(hass, entry)
    await agent.async_initialize()

    data = hass.data.setdefault(DOMAIN, {}).setdefault(entry.entry_id, {})
    data[CONF_API_KEY] = entry.data[CONF_API_KEY]
    data[DATA_AGENT] = agent

    conversation.async_set_agent(hass, entry, agent)
    return True

async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload OpenAI."""
    hass.data[DOMAIN].pop(entry.entry_id)
    conversation.async_unset_agent(hass, entry)
    return True

class HAFunctionCallbackHandler(AsyncCallbackHandler):
    """Callback handler for function execution in Home Assistant context."""
    
    def __init__(self, hass: HomeAssistant, agent: "OpenAIAgent"):
        self.hass = hass
        self.agent = agent
        
    async def on_tool_start(self, tool_name: str, tool_input: str, **kwargs: Any) -> None:
        """Handle tool start."""
        _LOGGER.debug("Starting tool %s with input %s", tool_name, tool_input)
        
    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Handle tool end."""
        _LOGGER.debug("Tool finished with output: %s", output)

class OpenAIAgent(conversation.AbstractConversationAgent):
    """OpenAI conversation agent using LangChain."""

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the agent."""
        self.hass = hass
        self.entry = entry
        self.history: dict[str, ConversationBufferMemory] = {}
        self.llm = None
        self.tools = []
        self.agent_executor = None

    async def async_initialize(self) -> None:
        """Initialize the LangChain components."""
        base_url = self.entry.data.get(CONF_BASE_URL)
        model = self.entry.options.get(CONF_CHAT_MODEL, DEFAULT_CHAT_MODEL)
        temperature = self.entry.options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)
        max_tokens = self.entry.options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        top_p = self.entry.options.get(CONF_TOP_P, DEFAULT_TOP_P)

        llm_kwargs = {
            "model_name": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "streaming": False,
            "verbose": True,
        }

        if is_azure(base_url):
            self.llm = AzureChatOpenAI(
                azure_endpoint=base_url,
                openai_api_version=self.entry.data.get(CONF_API_VERSION),
                openai_api_key=self.entry.data[CONF_API_KEY],
                **llm_kwargs,
            )
        else:
            self.llm = ChatOpenAI(
                openai_api_key=self.entry.data[CONF_API_KEY],
                base_url=base_url,
                organization=self.entry.data.get(CONF_ORGANIZATION),
                **llm_kwargs,
            )

        # Initialize tools from functions
        self.tools = await self._setup_tools()

    async def _setup_tools(self) -> list[StructuredTool]:
        """Set up LangChain tools from functions configuration."""
        functions = self.get_functions()
        tools = []
        
        for func in functions:
            executor = get_function_executor(func["function"]["type"])
            tool = StructuredTool.from_function(
                name=func["spec"]["name"],
                description=func["spec"]["description"],
                func=lambda **kwargs: executor.execute(
                    self.hass, func["function"], kwargs, None, self.get_exposed_entities()
                ),
                args_schema=func["spec"]["parameters"],
            )
            tools.append(tool)
            
        return tools

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    def get_exposed_entities(self):
        """Get exposed entities with their states and attributes."""
        states = [
            state
            for state in self.hass.states.async_all()
            if async_should_expose(self.hass, conversation.DOMAIN, state.entity_id)
        ]
        entity_registry = er.async_get(self.hass)
        exposed_entities = []
        
        for state in states:
            entity_id = state.entity_id
            entity = entity_registry.async_get(entity_id)
            
            aliases = []
            if entity and entity.aliases:
                aliases = entity.aliases

            exposed_entities.append(
                {
                    "entity_id": entity_id,
                    "name": state.name,
                    "state": self.hass.states.get(entity_id).state,
                    "aliases": aliases,
                    "attributes": str(state.attributes)
                }
            )
        return exposed_entities

    def get_functions(self):
        """Get functions configuration."""
        try:
            function = self.entry.options.get(CONF_FUNCTIONS)
            result = yaml.safe_load(function) if function else DEFAULT_CONF_FUNCTIONS
            if result:
                for setting in result:
                    function_executor = get_function_executor(
                        setting["function"]["type"]
                    )
                    setting["function"] = function_executor.to_arguments(
                        setting["function"]
                    )
            return result
        except (InvalidFunction, FunctionNotFound) as e:
            raise e
        except:
            raise FunctionLoadFailed()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        if user_input.conversation_id in self.history:
            conversation_id = user_input.conversation_id
            memory = self.history[conversation_id]
        else:
            conversation_id = ulid.ulid()
            user_input.conversation_id = conversation_id
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            self.history[conversation_id] = memory

            # Set initial system message
            try:
                system_prompt = self._generate_system_message(
                    self.get_exposed_entities(), user_input
                )
                memory.chat_memory.add_message(SystemMessage(content=system_prompt))
            except TemplateError as err:
                _LOGGER.error("Error rendering prompt: %s", err)
                intent_response = intent.IntentResponse(language=user_input.language)
                intent_response.async_set_error(
                    intent.IntentResponseErrorCode.UNKNOWN,
                    f"Sorry, I had a problem with my template: {err}",
                )
                return conversation.ConversationResult(
                    response=intent_response, conversation_id=conversation_id
                )

        # Create agent with tools and memory
        agent = OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm,
            tools=self.tools,
            memory=memory,
        )

        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=memory,
            callbacks=[HAFunctionCallbackHandler(self.hass, self)],
            verbose=True,
        )

        try:
            # Execute agent
            result = await agent_executor.arun(
                input=user_input.text,
                chat_history=memory.chat_memory.messages,
            )

            # Fire event
            self.hass.bus.async_fire(
                EVENT_CONVERSATION_FINISHED,
                {
                    "response": result,
                    "user_input": user_input,
                    "messages": memory.chat_memory.messages,
                },
            )

            # Create response
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_speech(result)
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

        except Exception as err:
            _LOGGER.error("Error during conversation: %s", err)
            intent_response = intent.IntentResponse(language=user_input.language)
            intent_response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                f"Sorry, I encountered an error: {err}",
            )
            return conversation.ConversationResult(
                response=intent_response,
                conversation_id=conversation_id,
            )

    def _generate_system_message(
        self, exposed_entities, user_input: conversation.ConversationInput
    ) -> str:
        """Generate the system message."""
        raw_prompt = self.entry.options.get(CONF_PROMPT, DEFAULT_PROMPT)
        return template.Template(raw_prompt, self.hass).async_render(
            {
                "ha_name": self.hass.config.location_name,
                "exposed_entities": exposed_entities,
                "current_device_id": user_input.device_id,
            },
            parse_result=False,
        )

import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample.json
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")

config_list_gemini = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gemini-1.5-pro-latest"],
    },

)
config_list_llama3 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["llama3-70b-8192"],
    },
)
config_list_mixtral = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["mixtral-8x7b-32768"],
    },
)
config_list_gpt = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-3.5-turbo"],
    },
)
config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4"],
    },
)

seed = 25  # for caching

assistant = AssistantAgent("assistant-pro", llm_config={"config_list": config_list_gemini})
user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding", "use_docker": False})
#user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change from Jan.2022 to Mar. 2024.")
# This initiates an automated chat between the two agents to solve the task

#user_proxy.initiate_chat(assistant, message="""创建一个线性回归模型，对data.csv进行回归训练，其中'C'栏为回归目标，
#                         然后调用shap.summary_plot针对测试数据画小提琴图展示,运行成功后保存代码""")
#result = user_proxy.initiate_chat(assistant, message="write python program to create a transformer model based on tensorflow and test it with some data,save your code to bo 'mymodel.py' , transformers package has been installed")
""" result = user_proxy.initiate_chat(assistant, message="write python program to use selenium in headless mode to grab some news from sohu.com, the selenium package has been installed,  
                                  the chrome is at /home/lxh/dev/withautogen/chrome-linux64/chrome,
                                  the chromedriver is at /home/lxh/dev/withautogen/chromedriver-linux64/chromedriver ")

 """

result = user_proxy.initiate_chat(assistant, message="写python程序使用pytorch对GPU进行压力测试，测试结果保存到文件中，pytorch库已经安装,首先检查GPU是否可用，确保使用gpu而且分配内存不超过4GB")



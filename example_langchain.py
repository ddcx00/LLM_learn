from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

template = """基于{product}的特性，写一段吸引人的广告文案。"""
prompt = PromptTemplate.from_template(template)

chain = LLMChain.from_string(prompt=prompt, llm="gpt-3.5-turbo")
print(chain.run(product="智能手表"))
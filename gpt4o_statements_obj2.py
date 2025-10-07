import pandas as pd
from openai import OpenAI
from tqdm import tqdm   

client = OpenAI(api_key="")

df = pd.read_csv("3countries_reddit.csv")  
df = df[df.source != 'human']

def build_prompt(question, country):
    return f"""
You are a culturally aware assistant from {country}.  
Given the question: {question}, generate a short conversational-style statement that someone might say, which could naturally lead another person to ask this question in the cultural context of {country}.  
Do not answer the question — only provide the preceding statement.  

Examples:
Country: Germany  
Question: "Is it acceptable to arrive 10 minutes late to a dinner party?"  
Statement: "I’m thinking of showing up a bit after the dinner party starts."  

Country: India  
Question: "Is it acceptable to eat with the left hand during a meal?"  
Statement: "I sometimes use my left hand when eating."  

Country: Egypt  
Question: "Is it polite to greet everyone individually at a gathering?"  
Statement: "Whenever I walk into a room, I like to greet each person one by one."  

Now generate the statement (in English) for:  
Country: {country}  
Question: {question}  

Statement:

"""

statements = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating initial statements"):
    country, question = row["country"], row["question"]
    prompt = build_prompt(question, country)
    response = client.chat.completions.create(
        model="gpt-4o", 
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=300
    )
    stmt = response.choices[0].message.content.strip()
    statements.append(stmt)

df["initial_statements"] = statements
df.to_csv("gpt4o_statements_reddit.csv", index=False)

print("\nSaved output to gpt4o_statements_reddit.csv")
from fastapi import FastAPI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import kdbai_client as kdbai
import pandas as pd
import io
from fastembed import TextEmbedding
import json
import matplotlib
import matplotlib.pyplot as plt
from fastapi.responses import StreamingResponse

matplotlib.use('Agg')  # Set the Matplotlib backend to Agg (needed for img bytes)

sa_model = f"cardiffnlp/twitter-roberta-base-sentiment"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session = kdbai.Session(endpoint='http://localhost:8082')

table = session.table('docTab')

model = SentenceTransformer('all-MiniLM-L6-v2')


BA_sentiments =  [
  {
    "label": "negative",
    "Date": "2020-04-29",
    "count": 4
  },
  {
    "label": "negative",
    "Date": "2023-07-26",
    "count": 2
  }
]



LHX_sentiments = [
    {"label":0,"Date":"2020-07-31","count":5.0},
    {"label":1,"Date":"2020-10-30","count":5.0},
    {"label":2,"Date":"2021-01-29","count":6.0},
    {"label":3,"Date":"2021-04-30","count":3.0},
    {"label":4,"Date":"2021-08-03","count":5.0}
]

GPT_DEPLOYMENT_NAME="gpt35turbo2"



def queryDoc(query, document="*", K=50):
    #K = 100
    qabot = RetrievalQA.from_chain_type(chain_type='stuff',
                                        llm = AzureChatOpenAI(
                                            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                            openai_api_version="2023-05-15",
                                            azure_deployment=GPT_DEPLOYMENT_NAME,
                                        ),
                                        retriever=vecdb_kdbai.as_retriever(search_kwargs=dict(k=K, filter=[['like', 'Document', str(document)]])),
                                        return_source_documents=True)
    ans = qabot.invoke(dict(query=query))
    return ans


def compareDocs(query, document1, document2):
    ans1 = queryDoc(query, document1)
    ans2 = queryDoc(query, document2)
    
    # Providing relevant context and original query for the model to generate a response
    llm_prompt = f"""Compare the two answers given to the question shown:
    Question:{query}
    Answer1:{ans1['result']}
    Answer2:{ans2['result']}
    Comparison:
    """
    
    client = AzureOpenAI(
        api_version="2023-05-15",
        azure_endpoint="https://uk-presales.openai.azure.com/",
    )

    response = client.chat.completions.create(
        model="gpt35turbo2",
        messages=[
            {'role': 'system', 'content': 'Your task is to compare the two answers given to the same question.'},
            {'role': 'user', 'content': llm_prompt},
        ],
        temperature=0.01,
    )
    
    return f"""Answer1 from {str(document1)}:
    {ans1['result']}
Answer2 from {str(document2)}:
    {ans2['result']}
Comparison:
    {response.choices[0].message.content}"""


@app.get("/stock-prices")
def get_stock_prices(sym: str, start_time: str, end_time: str):
    ba_prices_query = market_data_15.query(
        filter=[("=","sym",sym),("within","Date",[f"{start_time}", f"{end_time}"])],
    )
    ba_prices_df = pd.DataFrame(ba_prices_query)
    return ba_prices_df.to_dict(orient='records')

# Prompt 1: Search Boeing earnings calls from 2020 to 2022 for references to supply chain constraints (/transcripts)
@app.get("/transcripts")
def get_transcripts(sym: str, query: str, start_time: str = "2020-01-01", end_time: str = "2022-12-31"):
    # Convert the numpy array to a list using the correct method
    vector = next(embedding_model.embed([query])).tolist()
    transcripts_query = transcripts.search(
        filter=[("=","sym",'BA'),("within","Date",[f"{start_time}", f"{end_time}"])],
        aggs=['text', 'Date'],
        vectors=[vector],
        n=10
    )
    # Assuming transcripts.search returns data in a format that can be directly returned
    results = pd.DataFrame(transcripts_query[0])
    print(results)
    return results.to_dict(orient='records')
    # return sentimentAnalysis(results)


# Prompt 2: Display Boeing stock price data from 2020 to 2022 along with supply chain constraint sentiment (needs price)
@app.get("/plot/{sym}")
def plot_sentiment_and_price(sym: str, start_date: str = "2020-01-01", end_date: str = "2023-12-31"):
    price_data = session.table('market_data_15').query(filter=[("=", "sym", sym), ("within", "Date", [start_date, end_date])])

    sym_sentiments = BA_sentiments if sym == 'BA' else LHX_sentiments
    sentiment_df = pd.json_normalize(sym_sentiments)
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'])
    merged_df = price_data.merge(sentiment_df, on='Date', how='left')
    img = plot_price_sentiment_chart(merged_df)
    return StreamingResponse(img, media_type="image/png")

# compare prices
@app.get("/compare-prices")
def compare_prices(sym: str, date: str = "2020-10-28", subsector: bool = False):
    search_pattern = session.table('market_data_15').query(filter=[("=", "sym", "BA"), ("=", "Date", "2020-10-28")])['price'][0].tolist()
    price_data = session.table('market_data_15').query(aggs=["subsector", "sym"])
    search_filter = [("within", "Date", [date, str(pd.to_datetime(date) + pd.DateOffset(days=10))])]
    date_range = [date, str(pd.to_datetime(date) + pd.DateOffset(days=10))]
    num_return = 10
    if subsector:
        ad_syms = price_data[price_data['subsector'] == 'Aerospace & Defense']['sym'].unique().tolist()
        print(ad_syms)
        search_filter.append(("in", "sym", ad_syms))
        num_return = 5
    img =  execute_temporal_search_15(search_pattern, search_filter, num_return, date_range)
    return StreamingResponse(img, media_type="image/png")

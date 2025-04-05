from datetime import datetime
import pandas as pd
import openai
import json
import os
from main.database import data_retrieval
from loguru import logger
from  
# Initialize OpenAI API (Replace with your key)
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Instantiate the OpenAI client
client = openai.OpenAI()

def get_data(start_date, end_date):
    """Retrieves news data within the specified date range."""

    news_data = data_retrieval('news_sentiments')
    local_event_data = data_retrieval('local_events')
    global_event_data = data_retrieval('global_events')


    news_data_filtered = news_data[
        (pd.to_datetime(news_data['publish_date']) >= pd.to_datetime(start_date)) &
        (pd.to_datetime(news_data['publish_date']) <= pd.to_datetime(end_date))
    ]

    local_event_data_filtered = local_event_data[
        (pd.to_datetime(local_event_data['from_date']) >= pd.to_datetime(start_date)) &
        (pd.to_datetime(local_event_data['to_date']) <= pd.to_datetime(end_date))
    ]

    global_event_data_filtered = global_event_data[
        (pd.to_datetime(global_event_data['from_date']) >= pd.to_datetime(start_date)) &
        (pd.to_datetime(global_event_data['to_date']) <= pd.to_datetime(end_date))
    ]


    news_data_filtered.to_csv("news_data.csv")
    local_event_data_filtered.to_csv("local_events.csv")
    global_event_data_filtered.to_csv("global_events.csv")
    return news_data_filtered, local_event_data_filtered, global_event_data_filtered

def generate_summary(text_data):
    """Generates a summary of the provided text using OpenAI."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a expert data summarizer who reads different data from datatables, identifies crucial data facts and creates a perfect summary after analysing all the data in each column of the table."},
                {"role": "user", "content": f"Summarize the following text after analysing the text data. Enter all essiential data acquired from each column. Text data itself contains the data table. Summary should have a 2 sentenses for each row : {text_data}"}
            ],
            max_tokens=300,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None

def get_summary(start_date_str, end_date_str):
    """Retrieves news data, generates a summary, and saves the result to a JSON file."""
    try:
        # Parse the date strings
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        # Retrieve data within the date range
        news_data, local_events, global_events, customer_segment_data = get_data(start_date, end_date)
        # news_data = pd.read_csv('news_data.csv')
        news_data = news_data[['summary', 'impact', 'startDate', 'duration', 'reasoning', 'publish_date']]
        local_events_data = local_events.drop(['is_active', 'approval', 'approvedBy'], axis = 1)
        global_events_data = global_events.drop(['approval', 'approvedBy', 'createdBy'], axis =1)
        matrix_data = customer_segment_data.drop(['id'], axis = 1)
        
        print(f"news{news_data}")
        # Combine all text data for summary generation
        # all_text_data = " ".join(news_data['summary'].to_list())

        news_data['combined_text'] = news_data.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in news_data.columns]), axis=1)
        local_events_data['combined_text'] = local_events_data.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in local_events_data.columns]), axis=1)
        global_events_data['combined_text'] = global_events_data.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in global_events_data.columns]), axis=1)
        matrix_data['combined_text'] = matrix_data.apply(lambda row: ' '.join([f"{col}: {row[col]}" for col in matrix_data.columns]), axis=1)


        # Combine all rows into one large text block
        all_news_data = " ".join(news_data['combined_text'].to_list())
        all_local_events_data = " ".join(local_events_data['combined_text'].to_list())
        all_global_events_data = " ".join(global_events_data['combined_text'].to_list())
        all_matrix_data = " ".join(matrix_data['combined_text'].to_list())

        print(f"hhh{all_news_data}")
        # # Generate summary using OpenAI
        news_summary = generate_summary(all_news_data)
        time_series_summary = generate_summary(all_ts_data)
        local_events_summary = generate_summary(all_local_events_data)
        global_events_summary = generate_summary(all_global_events_data)
        matrix_summary = generate_summary(all_matrix_data)

        print(f"summary : {news_summary}")
    
        if news_summary and time_series_summary and local_events_summary and global_events_summary and matrix_summary:
            
            summary_data = {
                "news_data": news_summary.to_dict(orient='records'),
                "local_event_data" : local_events_summary.to_dict(orient='records'),
                "global_event_data" : global_events_summary.to_dict(orient='records'),
                "matrix_summary" : matrix_summary.to_dict(orient='records'),
                "time_series_summary" : time_series_summary.to_dict(orient='records')
            }

            with open("summary_result.json", "w") as f:
                json.dump(summary_data, f, indent=4)

            print("Summary generated successfully and saved to summary_result.json")
        else:
            print("Summary generation failed.")

        
        return news_summary, 
    except ValueError as ve:
        print(f"Date format error: {ve}")
    except FileNotFoundError:
        print("news_data.csv not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
# Example usage
if __name__ == "__main__":
    get_summary("2025-03-01", "2025-04-01")
    # _, _, loc = get_data("2025-03-01", "2025-04-01")
    # print(loc)
from flask import Flask, send_from_directory, render_template, request, redirect, url_for
from waitress import serve
from src.models.predictor import get_model
import pandas as pd

app = Flask(__name__, static_url_path="/static")

#input article options into this dataframe with same column names as how it started
options_df = pd.read_csv('app_test_df.csv')

#Comment out or use the following variable when testing app without running model for expediency sake.
OUTPUTS = ["WELLNESS", ' There are many studies on every aspect of happiness, the most coveted of all human achievements . Researchers define happiness as the ability to sustain an overall sense of well being over time . Happiness is not something to be earned, says author Sonja Lyubomirsky, Ph.D., author of The How of Happiness .']

#Based off of just headline of dropdown, it returns the full data for that article in pandas df to be sent to "run_model" function
def article_data_from_headline(headline, options_df):
    data = options_df[options_df['headline']==headline]
    return data

@app.route("/")
def index():
    """Return the main page."""
    return send_from_directory("static", "index.html")

@app.route("/run_model", methods=["POST"])
def run_model():
    """ Use the ML model to make a prediction using the form inputs. """

    # Convert the data into just a list of values to be sent to the model
    headline = request.form.get("article_headline")
    
    #get the whole row of just the headline you want and put it in data variable
    data = article_data_from_headline(headline, options_df)
    
    #from dataframe of 1, iloc and separate each thing needed
    authors = data['authors'].item()
    publish_date=data["date"].item()
    headline=data["headline"].item()
    article=data["article"].item()


    # This bit gets outputs from model or cheaty way from OUTPUTS var up top for expediency during testing
#     outputs = get_model(data)
    outputs = OUTPUTS
    
    
    print("within run_model:")
    print(outputs[0])
    print(outputs[1])
    print(authors)
    print(publish_date)
    print(headline)
    print(article)

    # Tell the browser to fetch the results page, passing along the prediction
    return redirect(url_for("show_results", topic = outputs[0], summary=outputs[1], article=article, headline=headline, publish_date=publish_date, authors=authors))

@app.route("/show_results")
def show_results():
    """ Display the results page with the provided prediction """
    
    # Extract the prediction from the URL params
    topic = request.args.get("topic")
    summary = request.args.get("summary")
    article = request.args.get("article")
    headline = request.args.get("headline")
    authors = request.args.get("authors")
    publish_date = request.args.get("publish_date")
    
    print("within show_results:")
    print(topic)
    print(summary)
    print(headline)
    print(authors)
    print(publish_date)
#     print(publish_date)
    print(article[:30])
    

    # Return the results pge
    return render_template("results.html", topic=topic, summary=summary, article=article, headline=headline, publish_date=publish_date, authors=authors)


if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=5000)

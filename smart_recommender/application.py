from flask import Flask
from flask import render_template
from flask import request
from nmf_recommender import nmf_recommendations


app = Flask(__name__)


@app.route('/')
def index():
     return render_template("index.html")

@app.route("/recommender")
def recommender():
    html_form_data = dict(request.args)


    recs = nmf_recommendations(html_form_data)

    return render_template("recommendations.html", movies=recs)



if __name__ == "__main__":
    app.run(host='0.0.0.0')

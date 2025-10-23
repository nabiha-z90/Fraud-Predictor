from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__, template_folder="templates")  # <-- make sure folder is correct

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        data = CustomData(
            step=int(request.form.get('step')),
            type=request.form.get('type'),
            amount=float(request.form.get('amount')),
            nameOrig=request.form.get('nameOrig'),
            oldbalanceOrg=float(request.form.get('oldbalanceOrg')),
            newbalanceOrig=float(request.form.get('newbalanceOrig')),
            nameDest=request.form.get('nameDest'),
            oldbalanceDest=float(request.form.get('oldbalanceDest')),
            newbalanceDest=float(request.form.get('newbalanceDest')),
        )
        df = data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        prediction = pipeline.predict(df)[0]

        result_text = "Fraud" if prediction == 1 else "Not Fraud"
        return render_template('home.html', results=result_text)
    
    except Exception as e:
        return render_template('home.html', results=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)

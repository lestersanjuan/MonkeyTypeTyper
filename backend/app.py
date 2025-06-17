from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/process', methods=['POST'])
def process():
    data = request.json
    text = data.get('text', '')
    result = text[::-1]  # Example: reverse the input string
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)

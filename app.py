from dash import Dash, dcc, html, Input, Output
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

app = Dash(__name__)

server = app.server

app.layout = html.Div([
    html.H1("Sentiment Analyzer Playground"),
    html.Div([
        "Input: ",
        dcc.Input(id='my-input', value='initial value', type='text')
    ]),
    html.Br(),
    html.Div(id='my-output'),

])


@app.callback(
    Output(component_id='my-output', component_property='children'),
    Input(component_id='my-input', component_property='value')
)
def update_output_div(input_value):
    result = classifier(input_value)[0]
    output_value = f"label: {result['label']}, "\
                   f"with score: {round(result['score'], 4)}"
    return f'Output: {output_value}'


if __name__ == '__main__':
    app.run_server(debug=True)

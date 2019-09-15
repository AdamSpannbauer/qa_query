# TODO: app isn't too pretty; do better
import dash
import dash_table
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

import ask_nasdaq as ask

external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    {
        'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
        'rel': 'stylesheet',
        'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
        'crossorigin': 'anonymous'
    }
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = 'Ask NASDAQ'
app.layout = html.Div([
    html.Div([
        html.Br(),
        html.H2('Ask NASDAQ Articles'),
        html.Hr(),
        html.Div([
            dcc.Input(id='question_input', value='Who is Tim Cook?', type='text',
                      style={'width': '30%'}),
            html.Button('Ask', id='ask_button', type='submit',
                        style={'margin-left': '10px'}),
        ], id='input')
    ]),
    html.Br(), html.Br(),
    html.Div(id='answers', style={'width': '80%'}),
], style={'margin-left': '10px'})


@app.callback(
    Output(component_id='answers', component_property='children'),
    [Input(component_id='ask_button', component_property='n_clicks')],
    state=[State(component_id='question_input', component_property='value')]
)
def ask_nasdaq(click, question):
    if click > 0:
        nasdaq_answer_df = ask.qa_query_nasdaq(question, QUERY_PARSER, SEARCHER, QA_BOT)
        nasdaq_answer_df = nasdaq_answer_df.sort_values('combined_rank')

        display_columns = [
            'answer_text',
            'answer_context',
            'source_title',
            # 'source_url',
        ]
        display_names = [
            'Answer',
            'Context',
            'Source Title',
            # 'Source URL',
        ]
        display_df = nasdaq_answer_df[display_columns]
        display_df.columns = display_names
        display_df = display_df.iloc[:4, :]

        datatable = dash_table.DataTable(
            id='answer_datatable',
            css=[{
                'selector': '.dash-cell div.dash-cell-value',
                'rule': 'display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;'
            }],
            columns=[{'name': i, 'id': i} for i in display_df.columns],
            data=display_df.to_dict('rows'),
        )

        return datatable


if __name__ == '__main__':
    search_fields = ['article_named_entities', 'article', 'title_named_entities', 'title']

    QA_BOT = ask.QABot(download=False)
    WHOOSH_IDX = ask.whoosh.index.open_dir('whoosh_idx', indexname='nasdaq')
    QUERY_PARSER = ask.MultifieldParser(search_fields,
                                        schema=WHOOSH_IDX.schema,
                                        group=ask.OrGroup)

    with WHOOSH_IDX.searcher() as SEARCHER:
        app.run_server()

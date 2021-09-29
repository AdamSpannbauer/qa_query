# TODO: app isn't too pretty; do better
import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output, State

import ask_nasdaq as ask

external_stylesheets = [
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    {
        "href": "https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css",
        "rel": "stylesheet",
        "integrity": "sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO",
        "crossorigin": "anonymous",
    },
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Ask NASDAQ"
app.layout = html.Div(
    [
        html.Div(
            [
                html.Br(),
                html.H2("Ask NASDAQ Articles"),
                html.Hr(),
                html.Div(
                    [
                        dcc.Input(
                            id="question_input",
                            value="Who is Tim Cook?",
                            type="text",
                            style={"width": "30%", "fontSize": 16},
                        ),
                        html.Button(
                            "Ask",
                            id="ask_button",
                            type="submit",
                            style={"margin-left": "10px"},
                        ),
                    ],
                    id="input",
                ),
            ]
        ),
        html.Br(),
        html.Br(),
        html.Div(id="answers", style={"width": "80%"}),
    ],
    style={"margin-left": "10px"},
)


@app.callback(
    Output(component_id="answers", component_property="children"),
    Input(component_id="ask_button", component_property="n_clicks"),
    State(component_id="question_input", component_property="value"),
)
def ask_nasdaq(click, question):
    if click and click > 0:
        display_fields = [
            "answer_text",
            "search_rank",
            "squad_rank",
            "combined_rank",
        ]
        display_names = [
            "Answer",
            "Search Rank",
            "Squad Rank",
            "Combined Rank",
        ]
        nasdaq_answer_df = ask.qa_query_nasdaq(
            question, QUERY_PARSER, SEARCHER, BERT_SQUAD
        )
        nasdaq_answer_df = ask.answer_display_df(
            nasdaq_answer_df,
            n_answers=10,
            display_fields=display_fields,
            display_names=display_names,
        )

        datatable = dash_table.DataTable(
            id="answer_datatable",
            css=[
                {
                    "selector": ".dash-cell div.dash-cell-value",
                    "rule": "display: inline; white-space: inherit; overflow: inherit; text-overflow: inherit;",
                }
            ],
            columns=[{"name": i, "id": i} for i in nasdaq_answer_df.columns],
            data=nasdaq_answer_df.to_dict("rows"),
            style_cell={"fontSize": 16},
        )

        return datatable


if __name__ == "__main__":
    SEARCH_TITLE = True
    SEARCH_ENTITIES = False

    if SEARCH_TITLE:
        search_fields = ["article"]
    else:
        search_fields = ["article", "title"]

    if SEARCH_ENTITIES:
        ent_search_fields = [f + "_named_entities" for f in search_fields]
        search_fields += ent_search_fields

    BERT_SQUAD = ask.BertSquad(download=False)
    WHOOSH_IDX = ask.whoosh.index.open_dir("whoosh_idx_ner", indexname="nasdaq")
    QUERY_PARSER = ask.MultifieldParser(
        search_fields, schema=WHOOSH_IDX.schema, group=ask.OrGroup
    )

    with WHOOSH_IDX.searcher() as SEARCHER:
        print("app will be running at http://localhost:8050/")
        app.run_server(debug=True)

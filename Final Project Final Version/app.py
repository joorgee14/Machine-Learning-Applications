import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
from datetime import datetime
from wordcloud import WordCloud
import io
import base64
import numpy as np
import spacy
from spacy.language import Language
from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel


# Load and preprocess data
questions = pd.read_csv("questions_complete.csv")
questions['CreationDate'] = pd.to_datetime(questions['CreationDate'])
questions['Date'] = questions['CreationDate'].dt.date
questions['Hour'] = questions['CreationDate'].dt.hour
questions['DayName'] = questions['CreationDate'].dt.day_name()
questions['YearMonth'] = questions['CreationDate'].dt.to_period('M').astype(str)
questions['Tags'] = questions['tag_pruned'].str.strip('[]').str.replace("'", "").str.split(', ')
tags_exploded = questions.explode('Tags')
tag_counts = tags_exploded['Tags'].value_counts().reset_index()
tag_counts.columns = ['Tag', 'Count']


# LDA Topic Modeling
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_md')

def normalize(doc):
    return [token.lemma_ for token in doc if token.has_vector and token.is_alpha and not token.is_punct and not token.is_stop and not token.ent_type_]

@Language.component("normalize_doc_component")
def create_normalize_doc_component(doc):
   return normalize(doc)

nlp.add_pipe("normalize_doc_component", after="ner")

# Process text
docs_fulltext = list(nlp.pipe(questions['joinedtext_clean'], batch_size=50))

# Create dictionary and filter
D = Dictionary(docs_fulltext)
lower_limit = 2
upper_limit = 0.80
D.filter_extremes(no_below=lower_limit, no_above=upper_limit)

# Create BOW corpus
bow = [D.doc2bow(doc) for doc in docs_fulltext]

# Train LDA model
num_topics = 10
ldag = LdaModel(corpus=bow, id2word=D, num_topics=num_topics)

# Get topic distributions
lda_vectors = []
for bow_doc in bow:
    topic_dist = ldag.get_document_topics(bow_doc, minimum_probability=0)
    lda_vectors.append([prob for _, prob in topic_dist])

lda_vectors = np.array(lda_vectors)
questions['DominantTopic'] = np.argmax(lda_vectors, axis=1)

# Get top words for each topic
topics_words = []
for i in range(num_topics):
    topic_words = ldag.show_topic(i, topn=10)
    topics_words.append(f"Topic {i}: " + ", ".join([word for word, prob in topic_words]))

questions['TopicDescription'] = questions['DominantTopic'].apply(lambda x: topics_words[x])

external_stylesheets = [
    "https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css",
]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "StackSample analysis"

app.layout = html.Div([
    html.Div([
        html.H1("StackSample Questions Classifier", 
               className="mb-3",
               style={'font-weight': 'bold', 
                     'color': '#2c3e50',
                     'text-align': 'left'})
    ], className="mb-4"),
    
    # Time Series Chart
    html.Div([
        html.H3("Question Activity Over Time", className="mb-3"),
        dcc.Dropdown(
            id='time-agg-dropdown',
            options=[
                {'label': 'Daily', 'value': 'D'},
                {'label': 'Weekly', 'value': 'W'},
                {'label': 'Monthly', 'value': 'M'}
            ],
            value='D',
            className="mb-3"
        ),
        dcc.Graph(id='questions-time-series',
                style={'height': '400px'})
    ], className="mb-4 p-3",
       style={'background-color': '#f8f9fa',
              'border-radius': '10px'}),
    
    # Tag Frequency Chart
    html.Div([
        html.H3("Question Frequency by Tag", className="mb-3"),
        dcc.Graph(id='tag-frequency-chart',
                style={'height': '500px'})
    ], className="mb-4 p-3",
       style={'background-color': '#f8f9fa',
              'border-radius': '10px'}),
    
    # Activity Heatmap
    html.Div([
        html.H3("Activity Heatmap", className="mb-3"),
        dcc.Graph(id='activity-heatmap',
                style={'height': '400px'})
    ], className="mb-4 p-3",
       style={'background-color': '#f8f9fa',
              'border-radius': '10px'}),
    
    # LDA Topics Visualization
    html.Div([
        html.H3("Question Topics Analysis (LDA)", className="mb-3"),
        dcc.Dropdown(
            id='topic-selector',
            options=[{'label': topics_words[i], 'value': i} for i in range(num_topics)],
            value=0,
            className="mb-3"
        ),
        dcc.Graph(id='topic-scatter-plot',
                style={'height': '500px'}),
        dcc.Graph(id='topic-words-cloud',
                style={'height': '400px'})
    ], className="mb-4 p-3",
       style={'background-color': '#f8f9fa',
              'border-radius': '10px'})
], className="container-fluid", style={'padding': '20px'})

# Callbacks
@app.callback(
    Output('questions-time-series', 'figure'),
    Input('time-agg-dropdown', 'value')
)
def update_time_series(time_agg):
    if time_agg == 'D':   # Daily
        time_series = questions.groupby('Date').size().reset_index(name='Count')
        fig = px.line(time_series, x='Date', y='Count')
    elif time_agg == 'W':   # Weekly
        time_series = questions.groupby(pd.Grouper(key='CreationDate', freq='W-MON')).size().reset_index(name='Count')
        fig = px.line(time_series, x='CreationDate', y='Count')
    else:   # Monthly
        time_series = questions.groupby('YearMonth').size().reset_index(name='Count')
        fig = px.line(time_series, x='YearMonth', y='Count')
    
    fig.update_layout(
        transition_duration=500,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@app.callback(
    Output('tag-frequency-chart', 'figure'),
    Input('time-agg-dropdown', 'value')
)
def update_tag_frequency_chart(_):
    fig = px.bar(tag_counts, 
                x='Tag', 
                y='Count',
                color='Count',
                color_continuous_scale='Viridis')
    
    fig.update_layout(
        xaxis_title='Tag',
        yaxis_title='Number of Questions',
        xaxis={'categoryorder':'total descending'},
        hovermode='x',
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Questions: %{y}<extra></extra>"
    )
    return fig

@app.callback(
    Output('activity-heatmap', 'figure'),
    Input('time-agg-dropdown', 'value')
)
def update_heatmap(_):
    heatmap_data = questions.groupby(['DayName', 'Hour']).size().reset_index(name='Count')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data['DayName'] = pd.Categorical(heatmap_data['DayName'], categories=day_order, ordered=True)
    heatmap_data = heatmap_data.sort_values('DayName')
    fig = px.density_heatmap(heatmap_data, x='Hour', y='DayName', z='Count',
                            color_continuous_scale='Viridis')
    fig.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@app.callback(
    [Output('topic-scatter-plot', 'figure'),
     Output('topic-words-cloud', 'figure')],
    Input('topic-selector', 'value')
)
def update_topic_visualizations(selected_topic):
    # Filter questions for selected topic
    topic_questions = questions[questions['DominantTopic'] == selected_topic]
    
    # 1. Scatter plot of Score vs Date
    scatter_fig = px.scatter(topic_questions, 
                           x='CreationDate', 
                           y='Score',
                           color='Score',
                           color_continuous_scale='Viridis',
                           title=f"Question Score Distribution Over Time",
                           hover_data=['Title'])
    
    scatter_fig.update_layout(
        xaxis_title='Creation Date',
        yaxis_title='Question Score',
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # 2. Word Cloud for Topic Words
    topic_words = [word for word, prob in ldag.show_topic(selected_topic, topn=50)]
    wordcloud = WordCloud(width=800, height=300,
                        background_color='white',
                        colormap='viridis').generate(' '.join(topic_words))
    img = io.BytesIO()
    wordcloud.to_image().save(img, 'PNG')
    img.seek(0)
    base64_img = base64.b64encode(img.getvalue()).decode('utf-8')
    img_src = f"data:image/png;base64,{base64_img}"
    
    words_fig = {
        'data': [],
        'layout': {
            'title': 'Top Topic Words',
            'images': [{
                'source': img_src,
                'xref': 'paper',
                'yref': 'paper',
                'x': 0.5,
                'y': 0.5,
                'sizex': 0.9,
                'sizey': 0.9,
                'xanchor': 'center',
                'yanchor': 'middle'
            }],
            'margin': dict(l=20, r=20, t=60, b=20),
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'paper_bgcolor': 'rgba(0,0,0,0)'
        }
    }
    
    return scatter_fig, words_fig

if __name__ == "__main__":
    app.run(debug=True)
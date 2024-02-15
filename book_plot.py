import plotly.express as px
import pandas as pd
import plotly

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('books.csv')

templates = ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]
# Create a bubble chart using Plotly Express
fig = px.scatter(df, x='x', y='y', color='labels', size='page_number', hover_data=['page_number', 'book_name', 'doc_class'],
                 labels={'page_number': 'Page Number', 'book_name': 'Book Name', 'doc_class': 'Document Class'},
                 title='Bubble Chart of Books', template=templates[3], color_continuous_scale=px.colors.sequential.Viridis)

# Show the plot
# fig.show()
fig.update_traces(marker=dict(line=dict(width=0.1, color='black'), sizeref=0.5))
plotly.offline.plot(fig)
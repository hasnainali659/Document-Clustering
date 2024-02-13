import plotly.express as px
import pandas as pd

# Load the CSV file into a Pandas DataFrame
df = pd.read_csv('books.csv')

# Create a bubble chart using Plotly Express
fig = px.scatter(df, x='x', y='y', color='labels', size='page_number', hover_data=['page_number', 'book_name'],
                 labels={'page_number': 'Page Number', 'book_name': 'Book Name'},
                 title='Bubble Chart of Books')

# Show the plot
fig.show()

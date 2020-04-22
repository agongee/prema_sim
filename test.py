import plotly.figure_factory as ff

df = [dict(Task="Job A", Start=0, Finish=10, Resource='Apple'),
      dict(Task="Job B", Start=10, Finish=24, Resource='Grape'),
      dict(Task="Job C", Start=24, Finish=36, Resource='Banana')]

colors = dict(Apple='rgb(220, 0, 0)', Grape='rgb(170, 14, 200)', Banana=(1, 0.9, 0.16))

fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True)
fig.show()
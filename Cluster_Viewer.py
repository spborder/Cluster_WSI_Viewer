"""

Testing out interactive graphing of Shriram's cluster metadata info

"""

import os
import sys
import pandas as pd
import numpy as np
import json

from glob import glob

try:
    import openslide
except:
    import tiffslide as openslide

from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

from dash import dcc, ctx, Dash
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, html, Input, Output, MultiplexerTransform, State

from timeit import default_timer as timer



def gen_layout(ftu_list,plot_types,labels):

    layout = html.Div([
        html.H1('Cluster WSI Region Viewer'),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    dbc.Card(
                        id = 'cluster-graph-card',
                        children = [
                            dbc.CardHeader('Cluster Graph'),
                            dbc.CardBody([
                                html.Div(
                                    dcc.Graph(id='cluster-graph',figure=go.Figure())
                                    )
                                ])
                            ]
                        )
                    ]),
                dbc.Col([
                    dbc.Card(
                        id='selected-image-card',
                        children = [
                            dbc.CardHeader('Selected Point'),
                            dbc.CardBody([
                                dcc.Graph(id='selected-image',figure=go.Figure()),
                                html.Div(id='selected-image-info')
                            ])
                        ]
                    )
                ]),
                dbc.Col([
                    dbc.Card(
                        id='plot-options',
                        children = [
                            dbc.CardHeader('Plot Options'),
                            dbc.CardBody([
                                dbc.Label('Functional Tissue Unit Type',html_for='ftu-select'),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div(
                                            dcc.Dropdown(
                                                ftu_list,
                                                ftu_list[0],
                                                id='ftu-select'
                                            )
                                        )
                                    )
                                ]),
                                html.B(),
                                dbc.Label('Type of plot',html_for='plot-select'),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div(
                                            dcc.Dropdown(
                                                plot_types,
                                                plot_types[0],
                                                id='plot-select'
                                            )
                                        )
                                    )
                                ]),
                                html.B(),
                                dbc.Label('Sample Labels',html_for='label-select'),
                                dbc.Row([
                                    dbc.Col(
                                        html.Div(
                                            dcc.Dropdown(
                                                labels,
                                                labels[0],
                                                id='label-select'
                                            )
                                        )
                                    )
                                ])
                            ])
                        ]
                    )
                ])
            ]),
        ])
    ])


    return layout

# Reading in the provided metadata
# Change path as necessary
base_dir = '/mnt/c/Users/Sam/Desktop/HIVE/SpotNet_NonEssential_Files/WSI_Heatmap_Viewer_App/assets/cluster_metadata/'
slide_dir = '/mnt/c/Users/Sam/Desktop/HIVE/FFPE/'

# Might have to change for different WSI formats
slide_names = glob(slide_dir+'*.svs')

# This part can be converted to just read_csv if that is what you have
# Converting JSON inputs into a dataframe for ease of reference
glom_metadata = json.load(open(base_dir+'FFPE_SpTx_Glomeruli.json'))
tub_metadata = json.load(open(base_dir+'FFPE_SpTx_Tubules.json'))

metadata = pd.DataFrame.from_dict(glom_metadata,orient='index')
metadata = pd.concat([metadata,pd.DataFrame.from_dict(tub_metadata,orient='index')],axis=0,ignore_index=True)

# Might only need one ftu and plot type but you can modify the lists below
ftu_list = ['glomerulus','Tubules']
plot_types = ['TSNE','UMAP']
labels = ['Cluster','image_id']

# Main metadata needed for each sample is:
#   - 'image_id' = slide name (There is a specific sub-string that is replaced with nothing below, you can remove that line)
#   - 'Min_x_coord','Min_y_coord','Max_x_coord','Max_y_coord' = bounding box coordinates at full resolution
#   - 'ftu_type' = this one can be removed if only using one ftu. Just remove the input from the first callback


class WSIClusterViewer:
    def __init__(self,
                app,
                layout,
                metadata,
                slide_list):

        self.app = app
        self.app.layout = layout
        self.app.title = 'Cluster Viewer App'
        self.metadata = metadata
        self.slide_list = slide_list

        self.app.callback(
            [Input('ftu-select','value'),
            Input('plot-select','value'),
            Input('label-select','value')],
            Output('cluster-graph','figure')
        )(self.update_graph)


        # You can change hoverData to clickData if that is less crazy
        self.app.callback(
            Input('cluster-graph','hoverData'),
            [Output('selected-image','figure'),
            Output('selected-image-info','children')]
        )(self.update_selected)


        # If you see OSError port is already in use or whatever, just change this port number
        self.app.run_server(debug=True,port=8070)


    def update_graph(self,ftu,plot,label):
        
        # Filtering by selected FTU
        current_data = self.metadata[self.metadata['ftu_type'].str.match(ftu)]

        if plot=='TSNE':
            plot_data_x = current_data['x_tsne'].tolist()
            plot_data_y = current_data['y_tsne'].tolist()

        elif plot=='UMAP':
            plot_data_x = current_data['x_umap'].tolist()
            plot_data_y = current_data['y_umap'].tolist()

        custom_data = list(current_data.index)
        label_data = current_data[label].tolist()

        graph_df = pd.DataFrame({'x':plot_data_x,'y':plot_data_y,'ID':custom_data,'Label':label_data})

        cluster_graph = px.scatter(graph_df,x='x',y='y',custom_data=['ID'],color='Label')


        return cluster_graph

    def grab_image(self,sample_info):

        slide_name = sample_info['image_id']

        # This is the line that removes the substring
        if type(slide_name)==str:
            slide_name = slide_name.replace('V10S15-103_','')
        else:
            slide_name = slide_name.tolist()[0].replace('V10S15-103_','')

        # openslide needs min_x, min_y, width, height
        min_x = int(sample_info['Min_x_coord'])
        min_y = int(sample_info['Min_y_coord'])
        width = int(sample_info['Max_x_coord'])-min_x
        height = int(sample_info['Max_y_coord'])-min_y
        
        slide_path = [i for i in self.slide_list if slide_name in i]

        slide_path = slide_path[0]

        wsi = openslide.OpenSlide(slide_path)
        slide_region = wsi.read_region((min_x,min_y),0,(width,height))
        openslide.OpenSlide.close(wsi)

        return np.uint8(np.array(slide_region))[:,:,0:3]
        

    def update_selected(self,hover):
        
        if hover is not None:
            sample_id = hover['points'][0]['customdata']
            sample_info = self.metadata.loc[sample_id]

        else:
            sample_info = self.metadata.iloc[0,:]

        selected_image = go.Figure(px.imshow(self.grab_image(sample_info)))
        image_metadata = json.dumps(sample_info.to_dict())


        return selected_image, image_metadata
    



def main(metadata,ftu_list,plot_types,labels,slide_names):
    external_stylesheets = [dbc.themes.LUX]
    app = DashProxy(__name__,external_stylesheets = external_stylesheets)
    layout = gen_layout(ftu_list,plot_types,labels)

    vis_cluster_app = WSIClusterViewer(app,layout,metadata,slide_names)


if __name__=='__main__':
    main(metadata,ftu_list,plot_types,labels,slide_names)















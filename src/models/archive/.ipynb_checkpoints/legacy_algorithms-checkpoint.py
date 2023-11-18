def LEGACY_make_graph_from_waypoints(self, min_passages=3):
    '''
    LEGACY function
    Transform computed waypoints to a weighted, directed graph
    The nodes of the graph are self.waypoints
    The edges are calculated by iterating through all significant points in each trajectory. 
    The significant points have been assigned a clusterID. Edges are added between pairwise clusters that follow each other 
    in the significant points dataframe.
    Example: The clusterID column of the significant point dataframe looks like this:
             1 1 1 4 4 5 5 7
             The algorithm extract the following edges from that:
             1-4, 4-5, 5-7
    Weakness: Clusters that are intersected by the actual trajectory are not added as edges. The function aggregate_edges()
    is built to rectify that, but is slow and does not catch all clusters.
    THIS ALGORITHM NEEDS TO BE IMPROVED
    '''     
    print(f'Constructing maritime traffic network graph from waypoints and trajectories...')
    start = time.time()  # start timer
    # create graph adjacency matrix
    n_clusters = len(self.waypoints)
    coord_dict = {}
    # for each trajectory, increase the weight of the adjacency matrix between two nodes
    for mmsi in self.significant_points.mmsi.unique():
        subset = self.significant_points[self.significant_points.mmsi == mmsi]
        subset = subset[subset.clusterID >=0]  # remove outliers
        if len(subset) > 1:  # subset needs to contain at least 2 waypoints
            for i in range(0, len(subset)-1):
                row = subset.clusterID.iloc[i]
                col = subset.clusterID.iloc[i+1]
                if row != col:  # no self loops
                    if (row, col) in coord_dict:
                        coord_dict[(row, col)] += 1  # increase the edge weight for each passage
                    else:
                        coord_dict[(row, col)] = 1  # create edge if it does not exist yet
    
    # store adjacency matrix as sparse matrix in COO format
    row_indices, col_indices = zip(*coord_dict.keys())
    values = list(coord_dict.values())
    A = coo_matrix((values, (row_indices, col_indices)), shape=(n_clusters, n_clusters))

    # Construct a GeoDataFrame from graph edges
    waypoints = self.waypoints
    waypoints.set_geometry('geometry', inplace=True, crs=self.crs)
    waypoint_connections = pd.DataFrame(columns=['from', 'to', 'geometry', 'direction', 'passages'])
    for orig, dest, weight in zip(A.row, A.col, A.data):
        # add linestring as edge
        p1 = waypoints[waypoints.clusterID == orig].geometry
        p2 = waypoints[waypoints.clusterID == dest].geometry
        edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
        # compute the orientation fo the edge (COG)
        p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
        p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
        direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
        line = pd.DataFrame([[orig, dest, edge, direction, weight]], 
                            columns=['from', 'to', 'geometry', 'direction', 'passages'])
        waypoint_connections = pd.concat([waypoint_connections, line])

    # Aggregate edges recursively
    # each edge that intersects the convex hull of another waypoint is divided in segments
    # the segments are added to the adjacency matrix and the original edge is deleted
    A_refined, waypoint_connections_refined, flag = geometry_utils.aggregate_edges(waypoints, waypoint_connections)
    while flag:
        A_refined, waypoint_connections_refined, flag = geometry_utils.aggregate_edges(waypoints, waypoint_connections_refined)
    
    # Construct a GeoDataFrame from graph edges
    waypoints = self.waypoints
    waypoints.set_geometry('geometry', inplace=True, crs=self.crs)
    waypoint_connections = pd.DataFrame(columns=['from', 'to', 'geometry', 'direction', 'passages'])
    for orig, dest, weight in zip(A_refined.row, A_refined.col, A_refined.data):
        # add linestring as edge
        p1 = waypoints[waypoints.clusterID == orig].geometry
        p2 = waypoints[waypoints.clusterID == dest].geometry
        edge = LineString([(p1.x, p1.y), (p2.x, p2.y)])
        # compute the orientation fo the edge (COG)
        p1 = Point(waypoints[waypoints.clusterID == orig].lon, waypoints[waypoints.clusterID == orig].lat)
        p2 = Point(waypoints[waypoints.clusterID == dest].lon, waypoints[waypoints.clusterID == dest].lat)
        direction = geometry_utils.calculate_initial_compass_bearing(p1, p2)
        line = pd.DataFrame([[orig, dest, edge, direction, weight]], 
                            columns=['from', 'to', 'geometry', 'direction', 'passages'])
        waypoint_connections = pd.concat([waypoint_connections, line])
    
    # initialize directed graph from adjacency matrix
    G = nx.from_scipy_sparse_array(A_refined, create_using=nx.DiGraph)

    # add node features
    for i in range(0, len(self.waypoints)):
        node_id = self.waypoints.clusterID.iloc[i]
        G.nodes[node_id]['n_members'] = self.waypoints.n_members.iloc[i]
        G.nodes[node_id]['position'] = (self.waypoints.lon.iloc[i], self.waypoints.lat.iloc[i])  # !changed lat-lon to lon-lat for plotting
        G.nodes[node_id]['speed'] = self.waypoints.speed.iloc[i]
        G.nodes[node_id]['cog_before'] = self.waypoints.cog_before.iloc[i]
        G.nodes[node_id]['cog_after'] = self.waypoints.cog_after.iloc[i]

    
    # report and save results
    print('------------------------')
    print(f'Unpruned Graph:')
    print(f'Number of nodes: {G.number_of_nodes()}')
    print(f'Number of edges: {G.number_of_edges()}')
    print('------------------------')
    self.G = G
    self.waypoint_connections = gpd.GeoDataFrame(waypoint_connections, geometry='geometry', crs=self.crs)

    # Prune network
    self.prune_graph(min_passages)
    
    end = time.time()  # end timer
    print(f'Time elapsed: {(end-start)/60:.2f} minutes')

def LEGACY_trajectory_to_path(self, trajectory):
    '''
    find the best path along the graph for a given trajectory and evaluate goodness of fit
    :param trajectory: a single MovingPandas Trajectory object
    '''
    G = self.G_pruned.copy()
    waypoints = self.waypoints.copy()
    connections = self.waypoint_connections_pruned.copy()
    points = trajectory.to_point_gdf()
    mmsi = points.mmsi.unique()[0]
    #print('=======================')
    #print(mmsi)
    #print('=======================')
    
    ### GET START POINT ###
    orig_WP, idx_orig, _ = geometry_utils.find_orig_WP(points, waypoints)
    
    ### GET END POINT ###
    dest_WP, idx_dest, _ = geometry_utils.find_dest_WP(points, waypoints)
    #print(orig_WP, dest_WP)
    
    try:
    # find all waypoints intersected by the trajectory
        #print('try')
        passages = geometry_utils.find_WP_intersections(trajectory, waypoints)
        if passages[0] != orig_WP:
            passages.insert(0, orig_WP)
        if passages[-1] != dest_WP:
            passages.append(dest_WP)
        if len(passages)<2:
            print('not enough intersections found')
            raise Exception
        #print(passages)
        # find edge sequence between each waypoint pair, that minimizes the distance between trajectory and edge sequence
        path = []
        for i in range(0, len(passages)-1):
            #min_sequence = nx.shortest_path(G, passages[i], passages[i+1])
            #edge_sequences = nx.all_simple_paths(G, passages[i], passages[i+1], cutoff=len(min_sequence)+1)
            edge_sequences = nx.all_shortest_paths(G, passages[i], passages[i+1])
            min_mean_distance = np.inf
            for edge_sequence in edge_sequences:
                # create a linestring from the edge sequence
                multi_line = []
                for j in range(0, len(edge_sequence)-1):
                    line = connections[(connections['from'] == edge_sequence[j]) & (connections['to'] == edge_sequence[j+1])].geometry.item()
                    multi_line.append(line)
                multi_line = MultiLineString(multi_line)
                multi_line = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                # measure distance between the multi_line and the trajectory
                WP1 = waypoints[waypoints.clusterID==edge_sequence[j]]['geometry'].item()
                WP2 = waypoints[waypoints.clusterID==edge_sequence[j+1]]['geometry'].item()
                idx1 = np.argmin(WP1.distance(points.geometry))
                idx2 = np.argmin(WP2.distance(points.geometry))
                if idx2 < idx1:
                    temp = idx1
                    idx1 = idx2
                    idx2 = temp
                eval_points = points.iloc[idx1:idx2+1]
                distances = eval_points.distance(multi_line)
                mean_distance = np.mean(distances)
                #print(edge_sequence)
                #print(mean_distance)
                if mean_distance < min_mean_distance:
                    min_mean_distance = mean_distance
                    best_sequence = edge_sequence
            path.append(best_sequence)
            #print('----------------------')
        flattened_path = [item for sublist in path for item in sublist]
        path = list(dict.fromkeys(flattened_path))
        message = 'success'
        #print('Found path:', path)

        path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
        for j in range(0, len(path)-1):
            #print(path[j], path[j+1])
            edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
            temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
            path_df = pd.concat([path_df, temp])
        path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)

        ###########
        # evaluate goodness of fit
        ###########
        eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
        multi_line = MultiLineString(list(path_df.geometry))
        edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
        distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
        mean_dist = distances.mean()
        median_dist = distances.median()
        max_dist = distances.max()
        evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                           'mean_dist':mean_dist,
                                           'median_dist':median_dist,
                                           'max_dist':max_dist,
                                           'distances':[distances.tolist()],
                                           'message':message}
                                         )
        #print(mmsi, ': success')
    
    except:
        if orig_WP == dest_WP:
            #print('origin is destination. Exiting...')
            message = 'orig_is_dest'
            path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
            evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                               'mean_dist':np.nan,
                                               'median_dist':np.nan,
                                               'max_dist':np.nan,
                                               'distances':[np.nan],
                                               'message':message}
                                             )
            #print(mmsi, ': orig_is_dest (no path)')
        
        else:
            try:
                #print('Attemptin shortest path method')
                path = nx.shortest_path(G, orig_WP, dest_WP)
                message = 'attempt'
                path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                for j in range(0, len(path)-1):
                    edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                    temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                    path_df = pd.concat([path_df, temp])
                path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
                ###########
                # evaluate goodness of fit
                ###########
                eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
                multi_line = MultiLineString(list(path_df.geometry))
                edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
                mean_dist = distances.mean()
                median_dist = distances.median()
                max_dist = distances.max()
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'mean_dist':mean_dist,
                                                   'median_dist':median_dist,
                                                   'max_dist':max_dist,
                                                   'distances':[distances.tolist()],
                                                   'message':message}
                                                 )
                #print(mmsi, ': attempt')
            except:
                message = 'failure'
                #print(mmsi, ': failure')
                path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'mean_dist':np.nan,
                                                   'median_dist':np.nan,
                                                   'max_dist':np.nan,
                                                   'distances':[np.nan],
                                                   'message':message}
                                                 )
    
    return path_df, evaluation_results

def LEGACY_trajectory_to_path(self, trajectory, verbose=False):
    '''
    Find the best path along the graph for a given trajectory and evaluate goodness of fit
    The algorithm contains the following steps:
    1. Find suitable waypoint close to the origin of the trajectory
    2. Find suitable waypoint close to the destination of the trajectory
    3. Find all waypoints passed by the trajectory
    :param trajectory: a single MovingPandas Trajectory object
    returns:
    :GeoDataFrame path_df: contains the sequence of edges traversed by a vessel which is closest to its original trajectory
    :DataFrame evaluation_results: contains metrics for the 'goodness of fit'
    '''
    G = self.G_pruned.copy()
    waypoints = self.waypoints.copy()
    connections = self.waypoint_connections_pruned.copy()
    points = trajectory.to_point_gdf()
    mmsi = points.mmsi.unique()[0]
    if verbose: 
        print('=======================')
        print(mmsi)
        print('=======================')
    
    ### GET potential START POINT ###
    orig_WP, idx_orig, dist_orig = geometry_utils.find_orig_WP(points, waypoints)
    
    ### GET potential END POINT ###
    dest_WP, idx_dest, dist_dest = geometry_utils.find_dest_WP(points, waypoints)
    #print('Potential start and end point:', orig_WP, dest_WP)

    ### GET ALL INTERSECTIONS between trajectory and waypoints
    passages = geometry_utils.find_WP_intersections(trajectory, waypoints)
    if verbose: print('Intersections found:', passages)
    
    # Distinguish three cases
    # 1. passages is empty and orig != dest
    if ((len(passages) == 0) & (orig_WP != dest_WP)):
        if dist_orig < 100:
            passages.append(orig_WP)
        if dist_dest < 100:
            passages.append(dest_WP)
    # 2. passages is empty and orig == dest --> nothing we can do here
    elif ((len(passages) == 0) & (orig_WP == dest_WP)):
        passages = []
    # 3. found passages
    else:
        # if the potential start waypoint is not in the list of intersections, but close to the origin of the trajectory, add it to the set of passages
        if ((orig_WP not in passages) & (dist_orig < 100) & (nx.has_path(G, orig_WP, passages[0]))):
            passages.insert(0, orig_WP)
        else:
            orig_WP = passages[0]
            orig_WP_point = waypoints[waypoints.clusterID==orig_WP]['geometry'].item()
            idx_orig = np.argmin(orig_WP_point.distance(points.geometry))
        
        # if the potential destination waypoint is not in the list of intersections, but close to the destination of the trajectory, add it to the set of passages
        if ((dest_WP not in passages) & (dist_dest < 100) & (nx.has_path(G, passages[-1], dest_WP))):
            passages.append(dest_WP)
        else:
            dest_WP = passages[-1]
            dest_WP_point = waypoints[waypoints.clusterID==dest_WP]['geometry'].item()
            idx_dest = np.argmin(dest_WP_point.distance(points.geometry))
    
    if len(passages) >= 2:
        try:
            if verbose: print('Executing try statement')
            path = []  # initialize the best edge sequence traversed by the vessel
            # find the edge sequence between each waypoint pair, that MINIMIZES THE DISTANCE between trajectory and edge sequence
            for i in range(0, len(passages)-1):
                # find all possible shortest connections between two waypoints
                # check if we are going backwards
                WP1 = waypoints[waypoints.clusterID==passages[i]]['geometry'].item()  # coordinates of waypoint at beginning of edge sequence
                WP2 = waypoints[waypoints.clusterID==passages[i+1]]['geometry'].item()  # coordinates of waypoint at end of edge sequence
                idx1 = np.argmin(WP1.distance(points.geometry))  # index of trajectory point closest to beginning of edge sequence
                idx2 = np.argmin(WP2.distance(points.geometry))  # index of trajectory point closest to end of edge sequence
                # if we are going backwards, skip to next WP
                if verbose: print('Point indices:', idx1, idx2)
                if verbose: print('From:', passages[i], ' To:', passages[i+1])
                if idx2 < idx1:
                    if verbose: print('going back is not allowed!')
                    passages[i+1]=passages[i]
                    continue
                if verbose: print('From:', passages[i], ' To:', passages[i+1])
                ### CORE FUNCTION
                # edge_sequences = nx.all_simple_paths(G, passages[i], passages[i+1], cutoff=5)  
                edge_sequences = nx.all_shortest_paths(G, passages[i], passages[i+1])
                # edge_sequences = nx.dijkstra_path(G, passages[i], passages[i+1], weight='inverse_weight')
                #################
                if verbose: print('=======================')
                if verbose: print(f'Iterating through edge sequences')
                min_mean_distance = np.inf
                # iterate over all possible shortest connections
                for edge_sequence in edge_sequences:
                    # create a linestring from the edge sequence
                    if verbose: print('Edge sequence:', edge_sequence)
                    multi_line = []
                    for j in range(0, len(edge_sequence)-1):
                        line = connections[(connections['from'] == edge_sequence[j]) & (connections['to'] == edge_sequence[j+1])].geometry.item()
                        multi_line.append(line)
                    multi_line = MultiLineString(multi_line)
                    multi_line = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                    # measure distance between the multi_line and the trajectory
                    WP1 = waypoints[waypoints.clusterID==edge_sequence[0]]['geometry'].item()  # coordinates of waypoint at beginning of edge sequence
                    WP2 = waypoints[waypoints.clusterID==edge_sequence[-1]]['geometry'].item()  # coordinates of waypoint at end of edge sequence
                    idx1 = np.argmin(WP1.distance(points.geometry))  # index of trajectory point closest to beginning of edge sequence
                    idx2 = np.argmin(WP2.distance(points.geometry))  # index of trajectory point closest to end of edge sequence
                    if verbose: print('Point indices:', idx1, idx2)
                    # Check if we are moving backwards
                    if idx2 < idx1:
                        if verbose: print('going back is not allowed! (inner)')
                        continue
                    eval_points = points.iloc[idx1:idx2+1]  # trajectory points associated with the edge sequence
                    distances = eval_points.distance(multi_line)  # distance between each trajectory point and the edge sequence
                    mean_distance = np.mean(distances)
                    ###### EXPERIMENTAL
                    if idx2>idx1:
                        sequence_length = multi_line.length
                        t1 = points.index[idx1]
                        t2 = points.index[idx2]
                        segment_length = trajectory.get_linestring_between(t1, t2).length
                        factor = sequence_length/segment_length
                        mean_distance = mean_distance*factor
                    ###############
                    if verbose: print('distances:', distances)
                    if verbose: print('Sequence:', edge_sequence)
                    if verbose: print('Mean distance:', mean_distance)
                    # when mean distance is smaller than any mean distance encoutered before --> save current edge sequence as best edge sequence
                    if mean_distance < min_mean_distance:
                        min_mean_distance = mean_distance
                        best_sequence = edge_sequence
                path.append(best_sequence)
                #print('----------------------')
            # delete duplicates from path
            flattened_path = [item for sublist in path for item in sublist]
            temp = [flattened_path[0]]
            for i in range(1, len(flattened_path)):
                if flattened_path[i] != flattened_path[i-1]:
                    temp.append(flattened_path[i])
            path = temp
            message = 'success'
            if verbose: print(mmsi, nx.is_path(G, path))
            if verbose: print('Found path:', path)
    
            # Compute GeoDataFrame from path, containing the edge sequence as LineStrings
            path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
            for j in range(0, len(path)-1):
                #print(path[j], path[j+1])
                edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                path_df = pd.concat([path_df, temp])
            path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
    
            ###########
            # evaluate goodness of fit
            ###########
            if idx_orig >= idx_dest:  # In some cases this is needed, for example for roundtrips of ferries
                idx_orig = 0
                idx_dest = -1
            eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
            multi_line = MultiLineString(list(path_df['geometry']))
            edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
            # compute the fraction of trajectory that can be associate with an edge sequence
            t1 = points.index[idx_orig]
            t2 = points.index[idx_dest]
            try:
                percentage_covered = trajectory.get_linestring_between(t1, t2).length / trajectory.get_length()
                length_ratio = edge_sequence.length / trajectory.get_linestring_between(t1, t2).length
            except:
                percentage_covered = 1
                length_ratio = edge_sequence.length / trajectory.get_length()
            #print('Length ratio:', length_ratio)
            distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
            # punishing 'wiggly' edge sequences
            if length_ratio > 1:
                distances = distances*length_ratio
            mean_dist = distances.mean()  # compute mean distance
            median_dist = distances.median()  # compute median distance
            max_dist = distances.max()  # compute max_distance
            evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                               'mean_dist':mean_dist,
                                               'median_dist':median_dist,
                                               'max_dist':max_dist,
                                               'distances':[distances.tolist()],
                                               'fraction_covered':percentage_covered,
                                               'length ratio': length_ratio,
                                               'message':message}
                                             )
            #print(mmsi, ': success')
            
        # In some cases the above algorithm gets stuck in waypoints without any connections leading to the next waypoint
        # In this case we attempt to find the shortest path between origin and destination
        except:
            if verbose: print('Executing except statement...')
            # if a path exists, we compute it
            if nx.has_path(G, passages[0], passages[-1]):
                path = nx.shortest_path(G, passages[0], passages[-1])
                message = 'attempt'
                # Compute GeoDataFrame from path, containing the edge sequence as LineStrings
                path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                for j in range(0, len(path)-1):
                    edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                    temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                    path_df = pd.concat([path_df, temp])
                path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
        
                ###########
                # evaluate goodness of fit
                ###########
                eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
                multi_line = MultiLineString(list(path_df.geometry))
                edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                distances = eval_points.distance(edge_sequence)  # compute distance between edge sequence and trajectory points
                mean_dist = distances.mean()
                median_dist = distances.median()
                max_dist = distances.max()
                t1 = points.index[idx_orig]
                t2 = points.index[idx_dest]
                try:
                    percentage_covered = trajectory.get_linestring_between(t1, t2).length / trajectory.get_length()
                except:
                    percentage_covered = 1
                #percentage_covered = len(eval_points) / len(points)
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'mean_dist':mean_dist,
                                                   'median_dist':median_dist,
                                                   'max_dist':max_dist,
                                                   'distances':[distances.tolist()],
                                                   'fraction_covered':percentage_covered,
                                                   'message':message}
                                                 )
            # If there is no path between origin and destination, we cannot map the trajectory to an edge sequence
            else:
                message = 'no_path'
                path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'mean_dist':np.nan,
                                                   'median_dist':np.nan,
                                                   'max_dist':np.nan,
                                                   'distances':[np.nan],
                                                   'fraction_covered':0,
                                                   'message':message}
                                                 )
    
    # When there are no intersections with any waypoints, we cannot map the trajectory to an edge sequence
    else:
        #print('Not enough intersections found. Cannot map trajectory to graph...')
        message = 'no_intersects'
        #print(mmsi, ': failure')
        path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
        evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                           'mean_dist':np.nan,
                                           'median_dist':np.nan,
                                           'max_dist':np.nan,
                                           'distances':[np.nan],
                                           'fraction_covered':0,
                                           'message':message}
                                         )
    return path_df, evaluation_results

def trajectory_to_path_sspd2(self, trajectory, verbose=False):
    '''
    Find the best path along the graph for a given trajectory and evaluate goodness of fit
    The algorithm contains the following steps:
    1. Find suitable waypoint close to the origin of the trajectory
    2. Find suitable waypoint close to the destination of the trajectory
    3. Find all waypoints passed by the trajectory
    :param trajectory: a single MovingPandas Trajectory object
    returns:
    :GeoDataFrame path_df: contains the sequence of edges traversed by a vessel which is closest to its original trajectory
    :DataFrame evaluation_results: contains metrics for the 'goodness of fit'
    '''
    G = self.G_pruned.copy()
    waypoints = self.waypoints.copy()
    connections = self.waypoint_connections_pruned.copy()
    points = trajectory.to_point_gdf()
    mmsi = points.mmsi.unique()[0]
    #print(mmsi)
    if verbose: 
        print('=======================')
        print(mmsi)
        print('=======================')
    
    ### GET potential START POINT ###
    orig_WP, idx_orig, dist_orig = geometry_utils.find_orig_WP(points, waypoints)
    
    ### GET potential END POINT ###
    dest_WP, idx_dest, dist_dest = geometry_utils.find_dest_WP(points, waypoints)
    if verbose: print('Potential start and end point:', orig_WP, dest_WP)

    ### GET ALL INTERSECTIONS between trajectory and waypoints and a channel around that trajectory that defines the subgraph
    passages, G_channel = geometry_utils.find_WP_intersections(points, trajectory, waypoints, G, 1000)
    if verbose: print('Intersections found:', passages)
    
    # Distinguish three cases
    # 1. passages is empty and orig != dest
    if ((len(passages) == 0) & (orig_WP != dest_WP)):
        if dist_orig < 100:
            passages.append(orig_WP)
        if dist_dest < 100:
            passages.append(dest_WP)
    # 2. passages is empty and orig == dest --> nothing we can do here
    elif ((len(passages) == 0) & (orig_WP == dest_WP)):
        passages = []
    # 3. found passages
    else:
        # if the potential start waypoint is not in the list of intersections, but close to the origin of the trajectory, add it to the set of passages
        if ((orig_WP not in passages) & (dist_orig < 100) & (nx.has_path(G, orig_WP, passages[0]))):
            passages.insert(0, orig_WP)
        else:
            orig_WP = passages[0]
            orig_WP_point = waypoints[waypoints.clusterID==orig_WP]['geometry'].item()
            idx_orig = np.argmin(orig_WP_point.distance(points.geometry))
        
        # if the potential destination waypoint is not in the list of intersections, but close to the destination of the trajectory, add it to the set of passages
        if ((dest_WP not in passages) & (dist_dest < 100) & (nx.has_path(G, passages[-1], dest_WP))):
            passages.append(dest_WP)
        else:
            dest_WP = passages[-1]
            dest_WP_point = waypoints[waypoints.clusterID==dest_WP]['geometry'].item()
            idx_dest = np.argmin(dest_WP_point.distance(points.geometry))
    if verbose: print('Intersections with start and endpoint:', passages)
    
    if len(passages) >= 2:
        #try:
        if verbose: print('Executing try statement')
        path = [[passages[0]]] # initialize the best edge sequence traversed by the vessel
        #eval_distances = []  # initialize list for distances between trajectory and edge sequence
        # find the edge sequence between each waypoint pair, that MINIMIZES THE DISTANCE between trajectory and edge sequence
        skipped = False
        for i in range(0, len(passages)-1):
            if verbose: print('From:', passages[i], ' To:', passages[i+1])
            if skipped == True:
                skipped = False
                continue
            WP1 = waypoints[waypoints.clusterID==passages[i]]['geometry'].item()  # coordinates of waypoint at beginning of edge sequence
            WP2 = waypoints[waypoints.clusterID==passages[i+1]]['geometry'].item()  # coordinates of waypoint at end of edge sequence
            idx1 = np.argmin(WP1.distance(points.geometry))  # index of trajectory point closest to beginning of edge sequence
            idx2 = np.argmin(WP2.distance(points.geometry))  # index of trajectory point closest to end of edge sequence
            if verbose: print('Point indices:', idx1, idx2)
            # Check if we are moving backwards
            if idx2 < idx1:
                if verbose: print('going back is not allowed! (inner)')
                continue
            ### CORE FUNCTION
            ### when current waypoint pair is very close, just take the shortest path to save computation time
            if (idx2-idx1) <= 3:
                if verbose: print('Close waypoints. Taking shortest path')
                edge_sequences = nx.all_shortest_paths(G_channel, passages[i], passages[i+1])
            # if waypoints are further apart, explore longer paths
            else:
                # compute length of shortest possible path
                min_sequence_length = len(nx.shortest_path(G_channel, passages[i], passages[i+1]))
                # if the shortest path is already long, just take the shortest path
                if min_sequence_length > 5:
                    if verbose: print('Far apart waypoints. Taking shortest path.')
                    edge_sequences = nx.all_shortest_paths(G_channel, passages[i], passages[i+1])
                # if the shortest path is short, explore alternative paths
                else:
                    if verbose: print('Far apart waypoints. Exploring all paths with limited length.')
                    cutoff = 5
                    edge_sequences = nx.all_simple_paths(G_channel, passages[i], passages[i+1], cutoff=cutoff)
                    while len(list(edge_sequences)) > 500:
                        cutoff -= 1
                        if verbose: print('Too many alternative paths. Reducing cutoff to ', cutoff)
                        edge_sequences = nx.all_simple_paths(G_channel, passages[i], passages[i+1], cutoff=cutoff)
                    if verbose: print('Final cutoff ', cutoff)
                    edge_sequences = nx.all_simple_paths(G_channel, passages[i], passages[i+1], cutoff=cutoff)    
            #################
            if idx2 == idx1:
                eval_points = points.iloc[idx1]  # trajectory points associated with the edge sequence
                eval_point = eval_points['geometry']
            else:
                eval_points = points.iloc[idx1:idx2+1]  # trajectory points associated with the edge sequence
                t1 = points.index[idx1]
                t2 = points.index[idx2]
                eval_traj = trajectory.get_linestring_between(t1, t2)  # trajectory associated with the edge sequence
                num_points = len(eval_points)
            if verbose: print('---------------------------')
            if verbose: print(f'Iterating through edge sequences')
            min_mean_distance = np.inf
            # iterate over all possible connections
            for edge_sequence in edge_sequences:
                # create a linestring from the edge sequence
                if verbose: print('Edge sequence:', edge_sequence)
                SSPD = geometry_utils.evaluate_edge_sequence(edge_sequence, connections, idx1, idx2, num_points, eval_traj, eval_points)
                if verbose: print('   SSPD:', SSPD)
                # when mean distance is smaller than any mean distance encountered before --> save current edge sequence as best edge sequence
                if SSPD < min_mean_distance:
                    min_mean_distance = SSPD
                    best_sequence = edge_sequence
            # sometimes we can skip a waypoint in between if it minimizes the SSPD
            if i < len(passages)-2:
                if nx.has_path(G_channel, passages[i], passages[i+2]):
                    WP1 = waypoints[waypoints.clusterID==passages[i]]['geometry'].item()  # coordinates of waypoint at beginning of edge sequence
                    WP2 = waypoints[waypoints.clusterID==passages[i+2]]['geometry'].item()  # coordinates of waypoint at end of edge sequence
                    idx1 = np.argmin(WP1.distance(points.geometry))  # index of trajectory point closest to beginning of edge sequence
                    idx2 = np.argmin(WP2.distance(points.geometry))  # index of trajectory point closest to end of edge sequence
                    edge_sequence = nx.shortest_path(G_channel, passages[i], passages[i+2])
                    eval_points = points.iloc[idx1:idx2+1]  # trajectory points associated with the edge sequence
                    t1 = points.index[idx1]
                    t2 = points.index[idx2]
                    eval_traj = trajectory.get_linestring_between(t1, t2)  # trajectory associated with the edge sequence
                    num_points = len(eval_points)
                    SSPD = geometry_utils.evaluate_edge_sequence(edge_sequence, connections, idx1, idx2, num_points, eval_traj, eval_points)
                    if SSPD < min_mean_distance:
                        if verbose: print(f'Hopping over waypoint {passages[i+1]}')
                        best_sequence = edge_sequence
                        skipped = True
            path.append(best_sequence[1:])
            if verbose: print(path)
            if verbose: print('=====================')
        # flatten path
        path = [item for sublist in path for item in sublist]
        #eval_distances = [item for sublist in eval_distances for item in sublist]
        message = 'success'
        if verbose: print('Found path:', path)
        if verbose: print(mmsi, nx.is_path(G, path))
        

        # Compute GeoDataFrame from path, containing the edge sequence as LineStrings
        path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
        for j in range(0, len(path)-1):
            #print(path[j], path[j+1])
            edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
            temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
            path_df = pd.concat([path_df, temp])
        path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)

        ###########
        # evaluate goodness of fit
        ###########
        if idx_orig >= idx_dest:  # In some cases this is needed, for example for roundtrips of ferries
            idx_orig = 0
            idx_dest = -1
        eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
        multi_line = MultiLineString(list(path_df['geometry']))
        edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
        # compute the fraction of trajectory that can be associated with an edge sequence
        t1 = points.index[idx_orig]
        t2 = points.index[idx_dest]
        try:
            eval_traj = trajectory.get_linestring_between(t1, t2)
            percentage_covered = eval_traj.length / trajectory.get_length()
        except:
            eval_traj = trajectory
            percentage_covered = 1
        num_points = len(eval_points)
        interpolated_points = [edge_sequence.interpolate(dist) for dist in range(0, int(edge_sequence.length)+1, int(edge_sequence.length/num_points)+1)]
        interpolated_points_coords = [Point(point.x, point.y) for point in interpolated_points]  # interpolated points on edge sequence
        interpolated_points = pd.DataFrame({'geometry': interpolated_points_coords})
        interpolated_points = gpd.GeoDataFrame(interpolated_points, geometry='geometry', crs=self.crs)    
        SSPD, d12, d21 = geometry_utils.sspd(eval_traj, eval_points['geometry'], edge_sequence, interpolated_points['geometry'])
        distances = d12.tolist() + d21.tolist()
        evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                           'SSPD':SSPD,
                                           'distances':[distances],
                                           'fraction_covered':percentage_covered,
                                           'message':message,
                                           'path':[path],
                                           'path_linestring':edge_sequence}
                                         )
        '''   
        # In some cases the above algorithm gets stuck in waypoints without any connections leading to the next waypoint
        # In this case we attempt to find the shortest path between origin and destination
        except:
            if verbose: print('Executing except statement...')
            # if a path exists, we compute it
            if nx.has_path(G, passages[0], passages[-1]):
                path = nx.shortest_path(G, passages[0], passages[-1])
                message = 'attempt'
                # Compute GeoDataFrame from path, containing the edge sequence as LineStrings
                path_df = pd.DataFrame(columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                for j in range(0, len(path)-1):
                    edge = connections[(connections['from'] == path[j]) & (connections['to'] == path[j+1])].geometry.item()
                    temp = pd.DataFrame([[mmsi, path[j], path[j+1], edge, message]], columns=['mmsi', 'orig', 'dest', 'geometry', 'message'])
                    path_df = pd.concat([path_df, temp])
                path_df = gpd.GeoDataFrame(path_df, geometry='geometry', crs=self.crs)
        
                ###########
                # evaluate goodness of fit
                ###########
                if idx_orig >= idx_dest:  # In some cases this is needed, for example for roundtrips of ferries
                    idx_orig = 0
                    idx_dest = -1
                eval_points = points.iloc[idx_orig:idx_dest]  # the subset of points we are evaluating against
                multi_line = MultiLineString(list(path_df.geometry))
                edge_sequence = ops.linemerge(multi_line)  # merge edge sequence to a single linestring
                t1 = points.index[idx_orig]
                t2 = points.index[idx_dest]
                try:
                    eval_traj = trajectory.get_linestring_between(t1, t2)
                    percentage_covered = eval_traj.length / trajectory.get_length()
                except:
                    eval_traj = trajectory
                    percentage_covered = 1
                num_points = len(eval_points)
                interpolated_points = [edge_sequence.interpolate(dist) for dist in range(0, int(edge_sequence.length)+1, int(edge_sequence.length/num_points)+1)]
                interpolated_points_coords = [Point(point.x, point.y) for point in interpolated_points]  # interpolated points on edge sequence
                interpolated_points = pd.DataFrame({'geometry': interpolated_points_coords})
                interpolated_points = gpd.GeoDataFrame(interpolated_points, geometry='geometry', crs=self.crs)    
                SSPD, d12, d21 = geometry_utils.sspd(eval_traj, eval_points['geometry'], edge_sequence, interpolated_points['geometry'])
                distances = d12.tolist() + d21.tolist()
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'SSPD':SSPD,
                                                   'distances':[distances],
                                                   'fraction_covered':percentage_covered,
                                                   'message':message,
                                                   'path':[path],
                                                   'path_linestring':edge_sequence}
                                                 )
            # If there is no path between origin and destination, we cannot map the trajectory to an edge sequence
            else:
                message = 'no_path'
                path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
                evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                                   'SSPD':np.nan,
                                                   'distances':[np.nan],
                                                   'fraction_covered':0,
                                                   'message':message,
                                                   'path':[np.nan],
                                                   'path_linestring':np.nan}
                                                 )
    '''
    # When there are no intersections with any waypoints, we cannot map the trajectory to an edge sequence
    else:
        #print('Not enough intersections found. Cannot map trajectory to graph...')
        message = 'no_intersects'
        #print(mmsi, ': failure')
        path_df = pd.DataFrame({'mmsi':mmsi, 'orig':orig_WP, 'dest':dest_WP, 'geometry':[], 'message':message})
        evaluation_results = pd.DataFrame({'mmsi':mmsi,
                                           'SSPD':np.nan,
                                           'distances':[np.nan],
                                           'fraction_covered':0,
                                           'message':message,
                                           'path':[np.nan],
                                           'path_linestring':np.nan}
                                         )
    return path_df, evaluation_results
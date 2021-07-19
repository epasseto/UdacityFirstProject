import math
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import matplotlib.style as mstyles
import matplotlib.pyplot as mpyplots
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import re
import seaborn as sns
from time import time

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_add_maintag(sub_columns, 
                  main_tag, 
                  verbose=False):
    '''This function adds the parent main tag to the names of the subcolumns
    Inputs:
      - sub_columns (mandatory) - - List
      - main_tag (mandatory) - - String
      - verbose (optional) - if you want some verbositiy - Boolean 
        (default=False)
    '''
    begin = time()
    better_tags = []
    
    for tag in sub_columns:
        better_tags.append(main_tag + '_' + tag)
    
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
    return better_tags

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_clear_maincol(name, 
                     sep_element='_',
                     give_child=False,
                     verbose=False):
    '''This function clears the name of a main label from the origin
    mail label is that super-column that organizes subcolumns
    as in Excel, you create in the first line "purchase" and in the second line 
    "influencer", "recommender", etc..
    Inputs:
      - name (mandatory) - a column name - String
      - sep_element (optional) - the separation element - String (default='_')
      - give child (optional) - if instead of giving the main column parent name,
        you want to retrieve the child subcolumn name, set it as True - 
        Boolean (default=False)
      - verbose (optional) - it is needed some verbosity to the process, turn 
        it on - Boolean (default=False)
    Output:
      - the column names before the LAST separation element
      *observe that subcolumns are kind of categorical type. So they couldn't 
       have a sepparation element in its name!
      **remember that we are trying to tokenize things, like categories...
    '''
    begin = time()
    pos_columns = []
    start_pos = 0
    position = False
    
    while True: #I know that this is risky... later I will insert a termination condition on my code!
        position = name.find(sep_element, start_pos) 
        
        if position == -1: #termination condition
            if not pos_columns:
                if verbose:
                    print('no division found')
                return False
            else:
                if verbose:
                    print('ending with positions:', pos_columns)
                last_position = pos_columns[-1] 
                end = time()
                if verbose:
                    print('elapsed time: {}s'.format(end-begin))
                if give_child:
                    return name[last_position+1:] #eliminate info from main column
                else:
                    return name[:last_position] #eliminate info from child subcolumn 
        else: #not yet!    
            pos_columns.append(position)
            start_pos = position + 1

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_clear_subcol(name, 
                    sep_element='_',
                    verbose=False):
    '''DEPRECATED! This function gives the name of a child column 
    Inputs:
      - name (mandatory) - the complete Column name from a well-formated Pandas 
        Dataset - String
      - sep_element (optional) - the sepparation element - String (default='_')
      - verbose (optional) - if you want some verbositiy - Boolean (default=False)
    Output:
      - only the name of the child column
    '''
    print('this function was deprecated! A fatal problem was found!')
    print('instead, use **fn_clear_maincol (..., give_child=True)') 
    #begin = time()
    #pos = name.find('_') #find main colum sepparation
    #end = time()
    #if verbose:
    #    print('elapsed time: {}s'.format(end-begin))
        
    return False #name[pos+1:] #eliminate info from parent main column (I don´t need it!) 

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_condense_subcols(col_series,
                        els_kinds,
                        sep_element=';',
                        verbose=False):
    '''This function transforms a row of a series of columns content into a 
    condensed form
    Input:
      - col_series (mandatory) - a series of data from a row of a dataset - 
        List from a Pandas Series
      - els_kinds (mandatory) - kinds of elements that I am dealing with - 
        List from harshed Child Column names
      - sep_element (optional) - a separation element for the condensing - 
        String (default=';')
      - verbose (optional) - if you want some verbosity - Boolean 
        (default=False)
    Output:
      - a string containing the condensed content
    *Use in this way - note that I need a dataset containing only child columns, 
     and take rows (axis=1): df_harsh_2011.apply(fn_condense_subcols, axis=1)
    '''
    print('WARNING: this function is deprecated! Instead, use **fn_condense_subcols()')
    return False

    #begin = time()
    #terms = []
    #if verbose:
    #    print('###Inputs###')
    #    print('col_series:', col_series)
    #    print('els_kinds:', els_kinds)
    #    print()
    #zipped_cols = zip(els_kinds, col_series)
    
    #for el_kind, position in zipped_cols:
    #    if verbose:
    #        print('testing for element kind: {} at position: {}'.format(el_kind, position))
            
    #    if fn_isNaN(position):
    #        if verbose:
    #            print('NaN found!')
    #    else:
    #        if verbose:
    #            print('element kind found:', el_kind)
    #        terms.append(el_kind)
            
    #source: https://realpython.com/python-string-split-concatenate-join/
    #entries = sep_element.join(terms)
    #if verbose:
    #    print('###Output###')
    #    print('entries condensed:', entries)
    
    #end = time()
    #if verbose:
    #    print('entries:', entries)
    #    print('elapsed time: {}s'.format(end-begin))
    
    #return entries

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_condense_subcols2(col_series,
                        els_kinds,
                        sep_element=';',
                        verbose=False):
    '''This function transforms a row of a series of columns content into a 
    condensed form
    Input:
      - col_series (mandatory) - a series of data from a row of a dataset - 
        List from a Pandas Series
      - els_kinds (mandatory) - kinds of elements that I am dealing with - 
        List from harshed Child Column names
      - sep_element (optional) - a separation element for the condensing - 
        String (default=';')
      - verbose (optional) - if you want some verbosity - Boolean 
        (default=False)
    Output:
      - a string containing the condensed content
    *Use in this way - note that I need a dataset containing only child columns, 
     and take rows (axis=1): df_harsh_2011.apply(fn_condense_subcols, axis=1)
    '''
    begin = time()
    terms = []
    if verbose:
        print('###Inputs###')
        print('col_series:', col_series)
        print('els_kinds:', els_kinds)
        print()
    
    filtered_cols = col_series[col_series.notna()]
    index_list = filtered_cols.index.tolist()
    for element in index_list:
        el_kind = fn_clear_maincol(element, give_child=True)
        if verbose:
            print('isolated element: {}'.format(el_kind))
        terms.append(el_kind)
    
    entries = sep_element.join(terms)#source: https://realpython.com/python-string-split-concatenate-join/
                
    if verbose:
        print('###Output###')
        print('entries condensed:', entries)
    
    end = time()
    if verbose:
        print('entries:', entries)
        print('elapsed time: {}s'.format(end-begin))
    
    return entries

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_cols(dataset_origin, 
                   dataset_destination, 
                   main_tag_origin,
                   main_tag_destination=False,  
                   sep_element=';', 
                   verbose=False):
    '''This function creates columns for a dataset
    Inputs:
      - dataset_origin (mandatory) - origin dataset - Pandas Dataset
      - dataset_destination (mandatory) - destination dataset - Pandas Dataset
      - main_tag_origin (mandatory) - main tag for origin - String
      - main_tag_destination (potional) - - String (default=False)
      - sep_element - separation element - String (default=';')
      - verbose (optional) - if you want some verbositiy - 
        Boolean (default=False)
    Output:
      - a dataset containing the created columns
    '''
    begin = time()
    if not main_tag_destination:
        main_tag_origin = main_tag_destination
        
    expanded_cols = fn_expand_col(dataset_destination=dataset_destination, 
                                  main_tag_destination=main_tag_destination, 
                                  verbose=verbose)
    
    basic_elements = {expanded_cols[i]: i for i in range(0, len(expanded_cols))}
    
    better_tags = fn_add_maintag(sub_columns=expanded_cols, 
                                 main_tag=main_tag_destination, 
                                 verbose=verbose)

    df_cols = pd.DataFrame(dataset_origin[main_tag_origin].apply(lambda x: fn_set_multicols(x, 
                                                                                            bas_elements=basic_elements, 
                                                                                            sep_element=sep_element, 
                                                                                            verbose=verbose)).tolist(),
                                                                                                columns=better_tags)
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))

    return df_cols

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_create_patches(bar_colors,
                      verbose=False):
    '''this function parses colors on bars
    Input:
      - colors on bar (mandatory) - Python list
    Output:
      - a tuple containing legends for eval() and commands for exec()'''
    legends = []
    commands = []
    
    dic_colors = {'b':('blue_patch', 'web languages'),
                  'r':('red_patch', 'scientific'),
                  'y':('yellow_patch', 'database'),
                  'g':('green_patch', 'shell')}

    for color in set(bar_colors):
        legends.append(dic_colors[color][0])
        command_leg = dic_colors[color][0] + " = mpatches.Patch(color='" + color + \
        "', label='" + dic_colors[color][1] + "')"
        if verbose:
            print(command_leg)
        commands.append(command_leg)
        
    #ordering and formatting for evaluation    
    legends = sorted(legends)
    legends = str(legends).replace("'","")

    return (legends, commands)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_enhance_cols(cols, 
                    general_dic={}, 
                    create_dic=False, 
                    main_label='Response', 
                    sub_label='Unnamed', 
                    verbose=False):
    '''This function takes double column names and reshape them into more 
    comprehensive single colum names
    It opperates in two ways:
    1-just send the double-column colum index and take a better column index or
    2-make two-steps correction 
      2.1.first run the function with create_dic=True, and get a dictionnary 
          with the captured actual column names
      2.2.then edit your *substitute* fields in your dic and save it into an 
          variable
      2.3.finally run again the function, feeding it with the names correction 
          dic, as general_dic=...
    Inputs:
      - cols (mandatory) - a double-row Pandas index
      - general_dic (optional) - column names for correction, only if dictionnary - 
        List (default={})
      - create_dic (optional) - a flag to produce a correction dictionnary - 
        Boolean (default=False)
      - main_label (optional) - the label that determines if first row is the 
        column name - String (default='Response')
      - sub_label (optional) -  the label says it inherits its name from a past 
        colum name - String (default='Unnamed')
      - verbose (optional) - if you need an explanatory verbosity, as the 
        function runs - Boolean (default=False)
    Possible Outputs:
      - a string with new better single-column names (create_dic=False)
      - a dictionnary with captured main-column names (create_dic=True)
    '''    
    begin = time()
    bad_columns = cols
    nice_columns = []
    text_add = False

    for col in bad_columns:
    
        if col[1] == main_label:
            if verbose:
                print('took a main label:', main_label)
            if create_dic: #only put the column name into a dictionnary for later substitution
                general_dic[col[0]] = '*substitute*'
            else: #if I already have a substitutive name in my dictionnary, use it!
                if col[0] in general_dic and general_dic[col[0]] != '*substitute*':
                    nice_col = general_dic[col[0]] #substitute it
                else:
                    nice_col = col[0] #take it
        else:
            #this is a subcolum for multicolumns
            nice_label = col[1].lower().replace("'", #made some corrections for column names
                                    "").replace(" ", 
                                    "").replace("(pleasespecify)", 
                                    "").replace(":", 
                                    "").replace(",", "")
            if col[0][:7] == sub_label:
                if not create_dic: #this section not apply for creating a substitution dictionnary!
                    if verbose:
                        print('*took a sub-label:{}...'.format(col[1][:8])) #show the first elements
                    if text_add: #OK to do the opperation
                        nice_col = text_add + '_' + nice_label 
                        if verbose:
                            print('*nice_col', nice_col)
                    else:
                        raise Exception('Error: subcolumn without a multicolumn was reached!') #something went wrong!
            else: #sub case test
                if verbose:
                    print('took a multi-label:{}...'.format(col[0][:15]))
                if create_dic: #only put the general column name into a dic for later substitution
                    general_dic[col[0]] = '*substitute*'
                else:
                    if col[0] in general_dic and general_dic[col[0]] != '*substitute*':
                        text_add = general_dic[col[0]]
                    else:
                        text_add = col[0] #take it for subs
                    nice_col = text_add + '_' + nice_label #for the first subcolumn
        
        if not create_dic:
            nice_columns.append(nice_col)

    end = time()
    
    if verbose:
        if create_dic:
            print('*************')
            print('general dictionnary created for later edition:')
            print(general_dic)
            print('*************')
            print('tip: cut & paste the dic and edit *substitute* for better column names!')
        else:
            print('*************')
            print('better columns names list created:')
            for column in nice_columns:
                print(column)
        print('*************')
        print('elapsed time: {}s'.format(end-begin) )
    
    if create_dic:
        return general_dic #substitution dic case
    else:
        return nice_columns #normal case

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_expand_col(dataset_destination, 
                  main_tag_destination, 
                  ending_cols=['nan','other'], 
                  verbose=False):
    '''This function gives a list of expanded columns from a dataset
    Inputs:
      - dataset_destination (mandatory)
      - main_tag_destination (mandatory)
      - ending_cols (optional) -  - List (default=['nan', 'other'])
      - verbose (optional) - if you want some verbosity - Boolean 
        (default=False)
    Output:
      - List of expanded columns
    '''
    begin = time()
    if verbose:
        print('taking element kinds using fn_take_child_cols')
    kinds = fn_take_child_cols(dataset=dataset_destination, 
                               parent_column=main_tag_destination, 
                               verbose=verbose)
    harsh_lst = []
    expanded_cols = []

    #first step create the basic subcolumns elements
    for kind in kinds:
        if verbose:
            print('clearing the element kind: {} using fn_clear_maincol(...give_child=True)'.format(kind))
        element = fn_clear_maincol(name=kind,
                                   give_child=True,
                                   verbose=verbose)
        harsh_lst.append(element)
        harsh = set(harsh_lst)    
    
    #second step 
    for element in sorted(harsh):
        if not element in ending_cols:
            expanded_cols.append(element)
    
    #if there are any of ending kinds
    if ending_cols:
        for ending_element in ending_cols:
            expanded_cols.append(ending_element)
            
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
    return expanded_cols

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_isNaN(num, 
             verbose=False):
    ''''This functions tests if a passed value is NaN
    according to specialists, NaN gives Falses when tested with theirselves!
    Input:
      - num (mandatory) - a Python Object
    Output:
      - a Boolean, indicating if is a NaN or not
    source: Towards Data Science - 5 Methods to Check NaN values in Python
    https://towardsdatascience.com/5-methods-to-check-for-nan-values-in-in-python-3f21ddd17eed
    '''
    if verbose:
        print('comparing NaN')
    return num!= num
              
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_plot_bars(languages,
                 bar_colors,
                 bars=7,
                 gra_style='dark_background',
                 txt_title='Languages Most Frequently Used',
                 verbose=False):
    '''This function generates barplots for plotting languages ranking
    Inputs:
      - languages (mandatory) - containing langugages data - Dictionnary
      - bar_colors (mandatory) - color classification for each bar - List
      - bars (optional) - number of bars to be plotted (must fit bar_colors length)
      - gra_style (optional) - plot style
      - txt_title (optional) - title for the graph
      - verbose (optional) - add some verbosity for the graph creation
    Output:
      - nothing
      '''
    language = []
    summation = []

    langs_zgen = languages[:bars]
    for langs, sums in langs_zgen:
        language.append(langs)
        summation.append(sums)
        
    if verbose:
        print('###languages barplot function initiated')

    mstyles.use(gra_style)    
    fig_zgen = mpyplots.figure() #creating the object
    axis_zgen = fig_zgen.add_axes([0,0,1,1]) #creating an axis

    #deactivated code (from Matplotlib documentation, for reference)
    #plt.subplot(211)
    #for cycling colors, use this
    #my_colors = list(islice(cycle(['b', 'r', 'g', 'y', 'k']), None, len(df)))
    #df.plot(kind='bar', stacked=True, color=my_colors)
    #fig_zgen.suptitle('Z-Generation referred programming languages')
    
    #generating the barplot
    axis_zgen.bar(language, summation, color=bar_colors )
    
    #later use autoformat for auto-rotating axis legends
    fig_zgen.autofmt_xdate()
    
    #creating patches for legends
    legends, commands = fn_create_patches(bar_colors=bar_colors, verbose=verbose)
    
    for command in commands:
        exec(command)
        
    #blue_patch = mpatches.Patch(color='b', label='web languages')
    #red_patch = mpatches.Patch(color='r', label='scientific')
    #yellow_patch = mpatches.Patch(color='y', label='database')

    axis_zgen.set_title(txt_title, fontsize=14)
    axis_zgen.legend(handles=eval(legends))#[blue_patch, red_patch, yellow_patch])#;
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_plot_bars2(languages1,
                  languages2,
                  bar_colors1,
                  bar_colors2,
                  bars=7,
                  gra_style='dark_background',
                  txt_title='Languages Most Frequently Used',
                  ax1_title='',
                  ax2_title='', 
                  verbose=False):
    '''This function generates barplots for plotting languages ranking
    Inputs:
      - languages (mandatory) - containing langugages data - Dictionnary
      - bar_colors (mandatory) - color classification for each bar - List
      - bars (optional) - number of bars to be plotted (must fit bar_colors length)
      - gra_style (optional) - plot style
      - txt_title (optional) - title for the graph
      - verbose (optional) - add some verbosity for the graph creation
    Output:
      - nothing
      '''
    if verbose:
        print('###languages barplot function initiated')
            
    #generating figure and both axis
    mstyles.use(gra_style)
    fig_zgen, (axis_zgen1, axis_zgen2) = mpyplots.subplots(1, 2, figsize=(12,6))

    #plitting data for plot
    language1, summation1 = fn_split_ls(languages1, bars=bars, verbose=verbose)
    language2, summation2 = fn_split_ls(languages2, bars=bars, verbose=verbose)
    
    #creating the two barplots
    axis_zgen1.bar(language1, summation1, color=bar_colors1)
    axis_zgen2.bar(language2, summation2, color=bar_colors2)

    #later use autoformat for auto-rotating axis legends
    fig_zgen.autofmt_xdate()

    if verbose:
        print('patches:')
        
    legends1, commands1 = fn_create_patches(bar_colors=bar_colors1, verbose=verbose)
    legends2, commands2 = fn_create_patches(bar_colors=bar_colors2, verbose=verbose)
    
    #creating the patches
    for command in set(commands1 + commands2):
        exec(command)
                    
    fig_zgen.suptitle(txt_title, fontsize=18)
    
    axis_zgen1.set_title(ax1_title, fontsize=12)
    axis_zgen2.set_title(ax2_title, fontsize=12)

    axis_zgen1.legend(handles=eval(legends1))
    axis_zgen2.legend(handles=eval(legends2))
    
#########1#########2#########3#########4#########5#########6#########7#########8
def fn_set_multicols(multiple, 
                     bas_elements, 
                     sep_element=';', 
                     verbose=False):
    '''This function returns a set of multicolumns for indexing 
    Inputs:
      - multiple (mandatory) - multiples - List
      - bas_elements (mandatory) - basic elements - List
      - sep_element - separation element - String (defalt=';')
      - verbose (optional) - if you want some verbositiy - Boolean 
        (default=False)
    Output:
      - a set of multicolumns
    '''
    begin = time()
    series_els = [0 for n in range (0, len(bas_elements))]

    try: #not NaN
        multiple_lst = set(multiple.lower().split(';'))
        diff_set = multiple_lst - set(bas_elements)
        if diff_set: #other case
            try:
                series_els[bas_elements['other']] = 1
            except KeyError:
                pass
        final_set = multiple_lst - diff_set
        for element in final_set:
            try:
                series_els[bas_elements[element]] = 1
            except KeyError:
                raise Exception('something went wrong: element must exist in dictionnary to be added!')
        if verbose:
            print(series_els)
    except AttributeError: #NaN case
        try:
            series_els[bas_elements['nan']] = 1
        except KeyError:
            pass
        end = time()
        if verbose:
            print('elapsed time: {}s'.format(end-begin))
            print(series_els)
            print('NaN')   

    return series_els

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_split_ls(languages,
                bars,
                verbose=False):
    '''this function splits languages dada
    Input:
      - languages (mandatory) - Python List
      - bars (mandatory) - number of bars desired
      - verbose (optional) - if you want some verbosity
    Output:
      - a tupple containing (languages, summation)    
    '''
    if verbose:
        print('split function started')

    language = []
    summation = []  
    langs_gen = languages[:bars]

    for langs, sums in langs_gen:
        language.append(langs)
        summation.append(sums)
        
    return (language, summation)

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_subcount_cols(col, 
                     sep_element=';',
                     spc_remove = False,
                     eliminate_empty=False,
                     verbose=False):
    '''This function takes a column that have multiple items, separate them and 
    count unique items for each registry.
    The objective is to count different individuals that are nested.
    It also returns the sum for NaN rows, it they exist.
    Inputs:
      - col (mandatory) - the column to be harshed - Pandas Series
      - sep_element (optional) - sepparation element - 
        String (optional, default=';')
      - spc_remove (optional) - if you want to remove also the internal spaces - 
        Boolean (optional, default=False)
      - verbose (optional) - it is needed some verbosity, turn it on - 
        Boolean (default=False)
    Output:
      - a Dictionnary with the counting for each item, plus the number of rows 
        with NaN
    '''
    begin = time()
    
    if eliminate_empty:
        col = col.replace(r'^\s*$', np.nan, regex=True) #just to elliminate empty rows
    
    items_dic = {'nan_rows': 0, #I already know that I want these entries, even if they finish as zero
                 'valid_rows': 0}
    harsh_dic = {} #temporary dictionnary for harshing
    nan_count = 0
    column = col
    
    for item in column:
        
        if fn_isNaN(item):
            if verbose:
                print('NaN')
            items_dic['nan_rows'] += 1
            nan_count += 1
        else:
            #It may be necessary to remove all spaces inside the harshed item
            #I found the best way to do this at Stack Overflow, here:
            #https://stackoverflow.com/questions/8270092/remove-all-whitespace-in-a-string
            if spc_remove:
                item = re.sub(r"\s+", "", item, flags=re.UNICODE) #credits: Stack Overflow
            else:
                item.strip() #only to ensure that I will not have blanks before and after the sentence
            
            small_col = set(item.lower().split(sep_element)) #I don´t want repeated items, so it is a set
            items_dic['valid_rows'] += 1 #add one item for valid rows
            if verbose:
                print('splitted registry:', small_col)
            
            for element in small_col:
                if verbose:
                    print('element for accounting:', element)
                if element in harsh_dic:
                    harsh_dic[element] += 1
                else:
                    harsh_dic[element] = 1

    #Why I made this strange sub-dictionnary insertion?
    #So, I think of a kind of Json structure will be really useful for my Udacity miniprojects
    #(Yes, I am really motivated to study Json... it looks nice for my programming future!)
    items_dic['elements'] = harsh_dic
    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        print('NaN total rows:', nan_count)
        print('*************')
        print('dictionnary of counting items created:')
        print(items_dic)
    
    return items_dic

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_take_child_cols(dataset, 
                       parent_column, 
                       verbose=False):
    '''This function takes all child columns from a Pandas Dataset, given a 
    parent Column name
    Inputs:
      - dataset (mandatory) - a well-formated - Pandas Dataset
      - parent_column (mandatory) - the name of the parent column - String
      - verbose (optional) - if you want messages during the processing - 
        Boolean (default=False)
    Output:
      - child-cols - a List, containing all the names of the child columns 
    '''    
    begin = time()
    child_cols = []

    for col_name in dataset.columns:
        if verbose:         
            print(col_name)
        if col_name[:len(parent_column)] == parent_column:
            if verbose:
                print('found a child:', col_name[len(parent_column)+1:])
            child_cols.append(col_name)

    return child_cols

#########1#########2#########3#########4#########5#########6#########7#########8
def fn_take_main_cols(dataset, verbose=False):
    '''This function harshes columns names from a dataset to find parent columns
    Input:
      - dataset (mandatory) - with formatted column names - Pandas Dataset
      - verbose - if you want messages during processing - Boolean 
        (default=False)
    Output:
      - a List with all the names of parent columns
    '''
    begin = time()
    main_columns = []
    last_column = 'False'
    not_added = True

    for column in dataset.columns:
        if verbose:
            print("tried:", column)

        dried_col_name = fn_clear_maincol(column)
        if dried_col_name:
            if verbose:
                print("candidate:", dried_col_name)
            if dried_col_name == last_column:
                if not_added:
                    main_columns.append(dried_col_name)
                    not_added = False
            else:
                last_column = dried_col_name
                not_added = True

    end = time()
    if verbose:
        print('elapsed time: {}s'.format(end-begin))
        
    return main_columns

#########1#########2#########3#########4#########5#########6#########7#########8
#########1#########2#########3#########4#########5#########6#########7#########8
def __main__():
  print('Priority one: insure return of organism for analysis.')
  print('All other considerations secondary. Crew expendable.')
  #12 useful functions in this package!
    
if __name__ == '__main__':
    main()
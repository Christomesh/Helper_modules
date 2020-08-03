def half_masked_corr_heatmap(dataframe,title=None,file=None):
    """
    dataframe: the refrence dataframe
    target: (string) column name of the target variable
    title: chart title
    file: pathfilename if you want to save image
    
    """
    plt.figure(figsize=(9,9))
    sns.set(font_scale=1)
    
    mask = np.zeros_like(dataframe.corr())
    mask[np.triu_indices_from(mask)] = True
    
    with sns.axes_style("white"):
        sns.heatmap(dataframe.corr(),mask=mask,annot=True,cmap='coolwarm')
    
    if title:
        plt.title(f"\n{title}",fontsize=18)
    
    if file:
        plt.savefig(file,bbox_inches='tight')
    
    return
    
    
def corr_to_target(dataframe,target,title=None,file=None):
    """
    dataframe: the refrence dataframe
    target: (string) column name of the target variable
    title: chart title
    file: pathfilename if you want to save image
    
    """
    plt.figure(figsize=(4,6))
    sns.set(font_scale=1)
    
    sns.heatmap(dataframe.corr()[[target]].sort_values(target,ascending=False),
               annot=True,cmap='coolwarm')
    if title:
        plt.title(f"{title}",fontsize=18)
    if file:
        plt.savefig(file,bbox_inches='tight')
    return


def gen_scatterplots(dataframe,target_column,list_columns,cols=1,file=None):
    import math
    """
    dataframe: the refrence dataframe
    target_columns: (string) column name of the target variable
    list_columns: list of columns to be used for scatter ploting
    cols : Number of plot you like to see in a row
    file: pathfilename if you want to save image
    
    """
    
    rows      = math.ceil(len(list_columns)/cols)
    figwidth  = 5 * cols
    figheight = 4 * rows
    
    fig,ax = plt.subplots(nrows  =rows,
                         ncols   =cols,
                         figsize =(figwidth,figheight))
    
    color_choices = ['blue','grey','goldenrod','r','black','darkorange','g']
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    ax = ax.ravel()
    
    for i,column in enumerate(list_columns):
        ax[i].scatter(dataframe[column],dataframe[target_column],
                     color = color_choices[i % len(color_choices)],
                     alpha = 0.25)
        ax[i].set_xlabel(f"{column}",fontsize=14)
        ax[i].set_ylabel(f"{target_column}",fontsize=14)
    
    fig.suptitle("\nScatter plot: Each feature vs Target",size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0,top=0.88)
    if file:
        plt.savefig(file,bbox_incehes='tight')
    plt.show()
    return

def gen_histograms(dataframe,bins=50,cols=1,file=None):
     
    """
    dataframe: the refrence dataframe
    cols : Number of plot you like to see in a row
    file: pathfilename if you want to save image
    
    """
    
    rows      = math.ceil(len(dataframe.columns)/cols)
    figwidth  = 5 * cols
    figheight = 4 * rows
    
    fig,ax = plt.subplots(nrows  =rows,
                         ncols   =cols,
                         figsize =(figwidth,figheight))
    
    color_choices = ['blue','grey','goldenrod','r','black','darkorange','g']
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    ax = ax.ravel()
    
    for i,column in enumerate(dataframe.columns):
        ax[i].hist(dataframe[column],bins=50,color=color_choices[i % len(color_choices)],
                  alpha=1)
        ax[i].set_title(f"{dataframe[column].name}",fontsize=18)
        
    fig.suptitle("\nHistograms for all variables in Dataframe",size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0,top=0.88)
    if file:
        plt.savefig(file,bbox_inches='tight')
    plt.show()
    
    return

def gen_boxplots(dataframe,cols=1,file=None):
      
    """
    dataframe: the refrence dataframe
    cols : Number of plot you like to see in a row
    file: pathfilename if you want to save image
    
    """
    
    rows      = math.ceil(len(dataframe.columns)/cols)
    figwidth  = 5 * cols
    figheight = 4 * rows
    
    fig,ax = plt.subplots(nrows  =rows,
                         ncols   =cols,
                         figsize =(figwidth,figheight))
    
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    ax = ax.ravel() # Ravel turns a matrix to a vector...easier to iterate
    
    for i,column in enumerate(dataframe.columns):
        ax[i].boxplot(dataframe[column])
        ax[i].set_title(f"{column}",fontsize=18)
        
    fig.suptitle("\nBoxplots for all variables in Dataframe",size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0,top=0.88)
    if file:
        plt.savefig(file,bbox_inches='tight')
    plt.show()
    
    return

def gen_linecharts(dataframe,cols=1,file=None):
      
    """
    dataframe: the refrence dataframe
    cols : Number of plot you like to see in a row
    file: pathfilename if you want to save image
    
    """
    
    rows      = math.ceil(len(dataframe.columns)/cols)
    figwidth  = 5 * cols
    figheight = 4 * rows
    
    fig,ax = plt.subplots(nrows  =rows,
                         ncols   =cols,
                         figsize =(figwidth,figheight))
    
    color_choices = ['blue','grey','goldenrod','r','black','darkorange','g']
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    ax = ax.ravel() # Ravel turns a matrix to a vector...easier to iterate
    
    for i,column in enumerate(dataframe.columns):
        ax[i].plot(dataframe[column],color=color_choices[i%len(color_choices)])
        ax[i].set_title(f"{column}",fontsize=18)
        ax[i].set_ylabel(f"{column}",fontsize=14)
        
    fig.suptitle("\nLine Graphs for all variables in Dataframe",size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0,top=0.88)
    if file:
        plt.savefig(file,bbox_inches='tight')
    plt.show()
    
    return

def gen_linecharts(dataframe,cols=1,file=None):
      
    """
    dataframe: the refrence dataframe
    cols : Number of plot you like to see in a row
    file: pathfilename if you want to save image
    
    """
    
    rows      = math.ceil(len(dataframe.columns)/cols)
    figwidth  = 5 * cols
    figheight = 4 * rows
    
    fig,ax = plt.subplots(nrows  =rows,
                         ncols   =cols,
                         figsize =(figwidth,figheight))
    
    color_choices = ['blue','grey','goldenrod','r','black','darkorange','g']
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    ax = ax.ravel() # Ravel turns a matrix to a vector...easier to iterate
    
    for i,column in enumerate(dataframe.columns):
        ax[i].plot(dataframe[column],color=color_choices[i%len(color_choices)])
        ax[i].set_title(f"{column}",fontsize=18)
        ax[i].set_ylabel(f"{column}",fontsize=14)
        
    fig.suptitle("\nLine Graphs for all variables in Dataframe",size=24)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0,top=0.88)
    if file:
        plt.savefig(file,bbox_inches='tight')
    plt.show()
    
    return
    
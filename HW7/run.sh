# /bin/sh

if [ $# -ge 1 ]; then

    if [ $1 = 'pcalda' ]; then 
        
        task=1
        python task1.py --task $task

        task=2
        python task1.py --task $task

        task=3
        kernel=1
        python task1.py --task $task --kernel $kernel

        task=3
        kernel=2
        python task1.py --task $task --kernel $kernel

        task=3
        kernel=3
        python task1.py --task $task --kernel $kernel
    
    elif [ $1 = 'tsne' ]; then

        mode=0
        perplexity=20
        python task2py --mode $mode --perplexity $perplexity

        mode=0
        perplexity=30
        python task2py --mode $mode --perplexity $perplexity

        mode=0
        perplexity=40
        python task2py --mode $mode --perplexity $perplexity

        mode=0
        perplexity=50
        python task2py --mode $mode --perplexity $perplexity

        mode=1
        perplexity=20
        python task2py --mode $mode --perplexity $perplexity

        mode=1
        perplexity=30
        python task2py --mode $mode --perplexity $perplexity

        mode=1
        perplexity=40
        python task2py --mode $mode --perplexity $perplexity

        mode=1
        perplexity=50
        python task2py --mode $mode --perplexity $perplexity

    else 

        echo "error argument $1 : function should be 'pcalda' or 'tsne'"

    fi
else
    echo 'args num should be >= 1'

fi
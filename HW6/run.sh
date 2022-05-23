#/bin/sh


if [ $# -eq 1 ]; then
   
    if [ $1 = 'one' ]; then

        # ctype [1:kkmeans, 2:spectral]
        # stype [1:ratio_cut, 2:normalized_cut]
        # itype [1:random, 2:kmeans++]
        gamma_c=0.0001
        gamma_s=0.0001
        k=3

        python clustering.py --ctype 1 --stype 1 --itype 1 --gamma_c $gamma_c --gamma_s $gamma_s --k $k 
        python clustering.py --ctype 2 --stype 1 --itype 2 --gamma_c $gamma_c --gamma_s $gamma_s --k $k 
        python clustering.py --ctype 2 --stype 2 --itype 2 --gamma_c $gamma_c --gamma_s $gamma_s --k $k 

    elif [ $1 = 'all' ]; then

        gamma_c=(0.0001 0.0001 0.001  0.001)
        gamma_s=(0.0001 0.001  0.0001 0.001)
        ks=(2 3 4)
        types=(1 2)

        len=${#gamma_c[@]}
        for k in ${ks[@]}; do
            for ((i=0;i<len;i++)); do
                for t in ${types[@]}; do

                    python clustering.py --ctype 1 --stype 1 --itype $t --gamma_c ${gamma_c[i]} --gamma_s ${gamma_s[i]} --k $k 
                    python clustering.py --ctype 2 --stype 1 --itype $t --gamma_c ${gamma_c[i]} --gamma_s ${gamma_s[i]} --k $k
                    python clustering.py --ctype 2 --stype 2 --itype $t --gamma_c ${gamma_c[i]} --gamma_s ${gamma_s[i]} --k $k
                
                done
            done
        done

    elif [ $1 = 'eg' ]; then

        gamma_c=(0.0001 0.00025  0.001)
        gamma_s=(0.0001 0.00025  0.001)

        for s in ${gamma_s[@]}; do
            for c in ${gamma_c[@]}; do

                time python clustering.py -seg -gs $s -gc $c
    
            done
        done

    else 
        
        echo "$1 : error function"

    fi

else 

    echo "should pass exact 1 argument"

fi
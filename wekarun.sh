#!/bin/bash
WEKALOC="weka.jar"
WEKA="java -Xmx1024m -classpath $WEKALOC"
DATADIR="data"
RESULTSDIR="results"
TRAINING_SETS=("heart_disease.arff")

cluster () {
    # Question 1
    echo "Clustering..."

    OUTPUTDIR=$RESULTSDIR/cluster
    mkdir -p $OUTPUTDIR
    find $OUTPUTDIR | grep ".txt" | xargs rm

    echo "Rows correspond to Cluster Algorithm k (1,2,3,4,6,9), Column correspond to number of K used (1,2,3,4,6,9)"
    for TRAINER in ${TRAINING_SETS[*]}; do
        echo $TRAINER
        echo "KMean-Manhattan"
        for n in 1 2 3 4 6 9; do
            for i in {1..10}; do
                $WEKA weka.clusterers.SimpleKMeans -N $n -A "weka.core.ManhattanDistance -R first-last" -I 500 -S $RANDOM -t $DATADIR/$TRAINER -c last 2>&1 > $OUTPUTDIR/MD-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR
            ruby script.rb
            cd ../..
            find $OUTPUTDIR | grep ".txt" | xargs rm
        done


        echo "KMean-Euclidean"
        for n in 1 2 3 4 6 9; do
            for i in {1..10}; do
                $WEKA weka.clusterers.SimpleKMeans -N $n -A "weka.core.EuclideanDistance -R first-last" -I 500 -S $RANDOM -t $DATADIR/$TRAINER -c last 2>&1 > $OUTPUTDIR/MD-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR
            ruby script.rb
            cd ../..
            find $OUTPUTDIR | grep ".txt" | xargs rm
        done

        echo "EM"
        for n in 1 2 3 4 6 9; do
            for i in {1..10}; do
                $WEKA weka.clusterers.EM -I 100 -N $n -M 1.0E-6 -S $RANDOM -t $DATADIR/$TRAINER -c last 2>&1 > $OUTPUTDIR/MD-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR
            ruby script.rb
            cd ../..
            find $OUTPUTDIR | grep ".txt" | xargs rm
        done
    done
}

reduce () {
    # Question 2 and 3
    echo "Reducing and clustering..."

    OUTPUTDIR=$RESULTSDIR/reduce

    mkdir -p $OUTPUTDIR/arff
    mkdir -p $OUTPUTDIR/kmeans


    for TRAINER in ${TRAINING_SETS[*]}; do
        echo $TRAINER

        echo "Random Projection, Rows correspond to dimensions reduced to: (1,2,3,4,6,9). Columns show repeated trials."

        for n in 1 2 3 4 6 9; do
            find $OUTPUTDIR | grep "\.arff\|\.txt" | xargs rm
            for i in {1..20}; do
                $WEKA weka.filters.unsupervised.attribute.RandomProjection -N $n -R $RANDOM -D Sparse1 -i $DATADIR/$TRAINER -o $OUTPUTDIR/arff/$TRAINER-n$n-i$i.arff -c last
                $WEKA weka.clusterers.SimpleKMeans -N $n -A "weka.core.ManhattanDistance -R first-last" -I 500 -S $RANDOM -t $OUTPUTDIR/arff/$TRAINER-n$n-i$i.arff -c last 2>&1 > $OUTPUTDIR/kmeans/RP-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR/kmeans
            ruby script.rb
            cd ../../..
        done

        return

        echo "PCA with n dimensions (1,2,3,4,5), kmeans"
        for n in 1 2 3 4 5; do
            find $OUTPUTDIR | grep "\.arff\|\.txt" | xargs rm
            for i in {1..5}; do
                $WEKA weka.filters.unsupervised.attribute.PrincipalComponents -R 0.95 -A 5 -M $n -i $DATADIR/$TRAINER -o $OUTPUTDIR/arff/$TRAINER-PCA-n$n-i$i.arff -c last
                $WEKA weka.clusterers.SimpleKMeans -N $n -A "weka.core.ManhattanDistance -R first-last" -I 500 -S $RANDOM -t $OUTPUTDIR/arff/$TRAINER-PCA-n$n-i$i.arff -c last 2>&1 > $OUTPUTDIR/kmeans/PCA-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR/kmeans
            ruby script.rb
            cd ../../..
        done
    done

}

nn_reduce () {
    # Question 4
    echo "Reducing and use Neural Network.."

    OUTPUTDIR=$RESULTSDIR/nn-reduce

    mkdir -p $OUTPUTDIR/arff
    mkdir -p $OUTPUTDIR/kmeans
    mkdir -p $OUTPUTDIR/nn

    echo "Rows correspond to dimensions reduced to: (1,2,3,4,6,9). Columns show repeated trials."

    for TRAINER in ${TRAINING_SETS[*]}; do
        echo $TRAINER
        echo "Random Projection"
        for n in 1 2 3 4 6 9; do
            find $OUTPUTDIR | grep "\.arff\|/.txt" | xargs rm
            for i in {1..2}; do
                $WEKA weka.filters.unsupervised.attribute.RandomProjection -N $n -R $RANDOM -D Sparse1 -i $DATADIR/$TRAINER -o $OUTPUTDIR/arff/$TRAINER-RP-n$n-i$i.arff -c last
                $WEKA weka.clusterers.SimpleKMeans -N $n -A "weka.core.ManhattanDistance -R first-last" -I 500 -S $RANDOM -t $OUTPUTDIR/arff/$TRAINER-RP-n$n-i$i.arff -c last 2>&1 > $OUTPUTDIR/kmeans/RP-$TRAINER-n$n-i$i.txt
                $WEKA weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S $RANDOM -E 20 -H a -R -t $OUTPUTDIR/arff/$TRAINER-RP-n$n-i$i.arff -c last 2>&1 > $OUTPUTDIR/nn/RP-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR/kmeans
            ruby script.rb
            cd ../../..

        done

        echo "PCA with n dimensions (1,2,3,4,5), kmeans and then nn"
        for n in 1 2 3 4 5; do
            echo $n
            find $OUTPUTDIR | grep "\.arff\|/.txt" | xargs rm
            for i in {1..5}; do
                $WEKA weka.filters.unsupervised.attribute.PrincipalComponents -R 0.95 -A 5 -M $n -i $DATADIR/$TRAINER -o $OUTPUTDIR/arff/$TRAINER-PCA-n$n.arff -c last
                $WEKA weka.clusterers.SimpleKMeans -N $n -A "weka.core.ManhattanDistance -R first-last" -I 500 -S $RANDOM -t $OUTPUTDIR/arff/$TRAINER-PCA-n$n.arff -c last 2>&1 > $OUTPUTDIR/kmeans/PCA-$TRAINER-n$n-i$i.txt
                $WEKA weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S $RANDOM -E 20 -H a -R -t $OUTPUTDIR/arff/$TRAINER-PCA-n$n.arff -c last 2>&1 > $OUTPUTDIR/nn/PCA-$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR/nn
            ruby script.rb
            cd ../../..

            cd $OUTPUTDIR/kmeans
            ruby script.rb
            cd ../../..
        done
    done
}

nn_orig () {
    echo "Training Neural Network on original data..."
    OUTPUTDIR=$RESULTSDIR/nn-orig
    mkdir -p $OUTPUTDIR/arff/unordered
    mkdir -p $OUTPUTDIR/nn

    # Question 5.5

    for TRAINER in ${TRAINING_SETS[*]}; do
        echo $TRAINER
        for n in 1; do
            find $OUTPUTDIR | grep "\.txt" | xargs rm
            for i in {1..10}; do
                $WEKA weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S $RANDOM -E 20 -H a -R -t $DATADIR/$TRAINER -c last 2>&1 > $OUTPUTDIR/nn/$TRAINER-n$n-i$i.txt
            done
            # echo $n

            cd $OUTPUTDIR/nn
            ruby script.rb
            cd ../../..
        done
    done
}

nn_cluster () {
    echo "Training neural network on data with extra clustered data attribute..."

    OUTPUTDIR=$RESULTSDIR/nn-cluster

    mkdir -p $RESULTSDIR/nn-cluster/arff/unordered
    mkdir -p $RESULTSDIR/nn-cluster/nn

    # Step 5

    echo "Rows correspond to k (1,2,3) clusters, columns are repeated trials."

    for TRAINER in ${TRAINING_SETS[*]}; do
        echo $TRAINER
        for n in 1 2 3; do
            find $OUTPUTDIR | grep "\.text\|\.arff" | xargs rm
            for i in {1..2}; do
                $WEKA weka.filters.unsupervised.attribute.AddCluster -W "weka.clusterers.SimpleKMeans -N $n -A \"weka.core.EuclideanDistance -R first-last\" -I 500 -S $RANDOM" -i $DATADIR/$TRAINER -o $OUTPUTDIR/arff/unordered/$TRAINER-n$n-i$i.arff -c last
                # $WEKA weka.filters.unsupervised.attribute.Reorder -R first-4,last,5 -i $OUTPUTDIR/arff/unordered/$TRAINER-n$n-i$i.arff -o $OUTPUTDIR/arff/$TRAINER-n$n-i$i.arff
                $WEKA weka.filters.unsupervised.attribute.Reorder -R first-5,last,6 -i $OUTPUTDIR/arff/unordered/$TRAINER-n$n-i$i.arff -o $OUTPUTDIR/arff/$TRAINER-n$n-i$i.arff
                $WEKA weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S $RANDOM -E 20 -H a -R -t $OUTPUTDIR/arff/$TRAINER-n$n-i$i.arff -c last 2>&1 > $OUTPUTDIR/nn/$TRAINER-n$n-i$i.txt
            done
            cd $OUTPUTDIR/nn
            ruby script.rb
            cd ../../..
        done
    done
}

usage () {
    echo "usage: $0 [cluster|reduce|nn_reduce|nn_cluster|nn_orig]"
    echo
    exit 1
}

if [[ $# -lt 1 ]]; then
    usage
fi

case $1 in
    cluster)
        cluster
        ;;
    reduce)
        reduce
        ;;
    nn_reduce)
        nn_reduce
        ;;
    nn_orig)
        nn_orig
        ;;
    nn_cluster)
        nn_cluster
        ;;
    all)
        cluster
        echo
        reduce
        echo
        nn_reduce
        echo
        nn_orig
        echo
        nn_cluster
        ;;
    *)
        usage
        ;;
esac
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            esac
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                fi

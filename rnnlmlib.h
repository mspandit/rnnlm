///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.4a
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
// (c) 2013 Cantab Research Ltd (info@cantabResearch.com)
//
///////////////////////////////////////////////////////////////////////

#ifndef _RNNLMLIB_H_
#define _RNNLMLIB_H_

const unsigned int PRIMES[]={108641969, 116049371, 125925907, 133333309, 145678979, 175308587, 197530793, 234567803, 251851741, 264197411, 330864029, 399999781,
407407183, 459258997, 479012069, 545678687, 560493491, 607407037, 629629243, 656789717, 716048933, 718518067, 725925469, 733332871, 753085943, 755555077,
782715551, 790122953, 812345159, 814814293, 893826581, 923456189, 940740127, 953085797, 985184539, 990122807};
const unsigned int PRIMES_SIZE=sizeof(PRIMES)/sizeof(PRIMES[0]);

const int MAX_NGRAM_ORDER=20;

enum FileTypeEnum {TEXT, BINARY, COMPRESSED};		//COMPRESSED not yet implemented

class CRnnLM{
protected:
    char train_file[MAX_STRING];
    char valid_file[MAX_STRING];
    char test_file[MAX_STRING];
    char rnnlm_file[MAX_STRING];
    char lmprob_file[MAX_STRING];
    
    int rand_seed;
    
    int debug_mode;
    
    int version;
    int filetype;
    
    int use_lmprob;
    real lambda;
    real gradient_cutoff;
    
    real dynamic;
    
    real alpha;
    real starting_alpha;
    int alpha_divide;
    double logp, llogp;
    float min_improvement;
    int iter;

	Vocabulary vocab;
	
    int train_words;
    int train_cur_pos;
    int counter;
    
    int one_iter;
    int maxIter;
    int anti_k;
    
    real beta;

	WordClass wordClass;
    int old_classes;

    long long direct_size;
    int direct_order;
    int history[MAX_NGRAM_ORDER];
	void direct_clearHistory();
	void direct_push(int);
	
	Backpropagation bp;
    
    int gen;

    int independent;
    
    LayerBackup layer0;		//neurons in input layer
    LayerBackup layer1;		//neurons in hidden layer
    LayerBackup layerc;		//neurons in hidden layer
    LayerBackup layer2;		//neurons in output layer
    
	MatrixBackup matrix01;
    MatrixBackup matrix12;		//weights between hidden and output layer (or hidden and compression if compression>0)
    MatrixBackup matrixc2;		//weights between hidden and compression layer

    direct_t *syn_d;		//direct parameters between input and output layer (similar to Maximum Entropy model parameters)
    direct_t *syn_db;
    
    
public:

    int alpha_set, train_file_set;

    CRnnLM()		//constructor initializes variables
    {
		version=10;
		filetype=TEXT;

		use_lmprob=0;
		lambda=0.75;
		gradient_cutoff=15;
		dynamic=0;

		train_file[0]=0;
		valid_file[0]=0;
		test_file[0]=0;
		rnnlm_file[0]=0;

		alpha_set=0;
		train_file_set=0;

		alpha=0.1;
		beta=0.0000001;
		//beta=0.00000;
		alpha_divide=0;
		logp=0;
		llogp=-100000000;
		iter=0;

		min_improvement=1.003;

		train_words=0;
		train_cur_pos=0;

		vocab.initialize(100, 0, 100000000);
		layer1.initialize(30);	
		layer1.clear();
		direct_size=0;
		direct_order=0;

		gen=0;

		independent=0;

		syn_d=NULL;
		syn_db=NULL;	
		//

		rand_seed=1;

		wordClass._size=100;
		old_classes=0;

		one_iter=0;
		maxIter=0;

		debug_mode=1;
		srand(rand_seed);

    }
    
    ~CRnnLM()		//destructor, deallocates memory
    {
		if (layer0._neurons != NULL) {	    
		    if (syn_d!=NULL) free(syn_d);

		    if (syn_db!=NULL) free(syn_db);
		}
    }
    
    void setTrainFile(char *str);
    void setValidFile(char *str);
    void setTestFile(char *str);
    void setRnnLMFile(char *str);
    void setLMProbFile(char *str) {strcpy(lmprob_file, str);}
    
    void setFileType(int newt) {filetype=newt;}
    
    void setClassSize(int newSize) {wordClass._size=newSize;}
    void setOldClasses(int newVal) {old_classes=newVal;}
    void setLambda(real newLambda) {lambda=newLambda;}
    void setGradientCutoff(real newGradient) {gradient_cutoff=newGradient;}
    void setDynamic(real newD) {dynamic=newD;}
    void setGen(real newGen) {gen=newGen;}
    void setIndependent(int newVal) {independent=newVal;}
    
    void setLearningRate(real newAlpha) {alpha=newAlpha;}
    void setRegularization(real newBeta) {beta=newBeta;}
    void setMinImprovement(real newMinImprovement) {min_improvement=newMinImprovement;}
    void setHiddenLayerSize(int newsize) {
		layer1.initialize(newsize);
		layer1.clear();
	}
    void setCompressionLayerSize(int newsize) {
		layerc.initialize(newsize);
		layerc.clear();
	}
    void setDirectSize(long long newsize) {direct_size=newsize;}
    void setDirectOrder(int newsize) {direct_order=newsize;}
    void setBPTT(int newval) {bp._bptt = newval;}
    void setBPTTBlock(int newval) {bp._block=newval;}
    void setRandSeed(int newSeed) {rand_seed=newSeed; srand(rand_seed);}
    void setDebugMode(int newDebug) {debug_mode=newDebug;}
    void setAntiKasparek(int newAnti) {anti_k=newAnti;}
    void setOneIter(int newOneIter) {one_iter=newOneIter;}
    void setMaxIter(int newMaxIter) {maxIter=newMaxIter;}
    

    int readWordIndex(FILE *fin);
    
    void saveWeights();			//saves current weights and unit activations
    void restoreWeights();		//restores current weights and unit activations from backup copy
    //void saveWeights2();		//allows 2. copy to be stored, useful for dynamic rescoring of nbest lists
    //void restoreWeights2();		
    void saveContext();
    void restoreContext();
    void saveContext2();
    void restoreContext2();
    void initialize();
    void saveNet();
    void goToDelimiter(int delim, FILE *fi);
    void restoreNet();
    void netFlush();
    void netReset();    //will erase just hidden layer state + bptt history + maxent history (called at end of sentences in the independent mode)

	void adjustWeights(int, int, int, real);
	void computeErrorVectors(int);

	void clearClassActivation(int);
	void normalizeOutputClassActivation();
	void layer2_normalizeActivation(int);
	
	void randomizeWeights(Synapse *, int, int);
    void computeProbDist(int last_word, int word);
    void learn(int last_word, int word);
    void copyHiddenLayerToInput();
    void trainNet();
    void useLMProb(int use) {use_lmprob=use;}
    void testNet();
    void testNbest();
    void testGen();

    void slowMatrixXvector(Neuron *dest, Neuron *srcvec, Synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type);    
    void matrixXvector(Layer &, Layer &, Matrix &, int matrix_width, int from, int to, int from2, int to2, int type);
	
	void layer2_clearActivation(Neuron *, int, int);
	void inputLayer_clear(Neuron [], int, int);
};

#endif

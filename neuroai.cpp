private:
        int* input;
        int* output;
        int* hiddeen;

        int* syn;
public:
        int* initinput(int numneurons);
        int* initoutuput(int numneurons);
        int* inithidden(int* numneuronlist, int numlayers);
        int* initsyn(int* matrixdimlist, int numlevels);


void initinput(int pnumneurons)
{
        input = (int*)malloc(pnumneurons*sizeof(int));
}

void initoutput(int pnumneurons)
{
        ouput = (int*)malloc(pnumneurons*sizeof(int);
}

void inithidden(int* pnumneuronlist, int pnumlayers)
{
        hidden = (int*)malloc(pnumlayers*sizeof(int*));
        for(int i = 0; i<pnumlayers; i++)
        {
                hidden[i] = (int*)malloc(pnumneuronlist[i]*sizeof(int));
        }
}

void initsyn(int* pmatrixdimlist, int pnumlevels)
{
        syn = (int*)malloc(pnumlevels*sizeof(int*));
        for(int i = 0; i<pnumlevels; i++)
        {
                syn[i] = (int*)malloc(pmatrixdimlist[i*2]*pmatrixdimlist[i*2+1]*sizeof(int));
        }
}

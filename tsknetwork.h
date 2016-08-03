#ifndef TSKNETWORK_H
#define TSKNETWORK_H


namespace TSKnetwork {

    int extract_float(char * s, float * k, int len);
    int extract_int(char *s, int * a, int len, int decr=0);
    void cout_matrix(float ** A, int m, int n);
    void solve(float ** T, int m, int n, int * indexes);

    class Neuron
    {
    public:
        virtual void setState()=0;
        float getState();
    protected:
        float State;
    };

    class InputNeuron: public Neuron
    {
    public:
        InputNeuron(float input_x=0, float min_domain=0, float max_domain=10);

     virtual void setState()  {  }
        void setState(float input_x);

    float a;
    float b;
    };

    class MuNeuron: public Neuron
    {
        friend class AgregationNeuron;
        friend class ConclusionNeuron;
    public:
        MuNeuron(InputNeuron * input, float B=1, float C=1, float SIGMA=3, float BIAS=0.00002);
        virtual void setState();
        float b;
        float c;
        float sigma;
        InputNeuron * Links; //array of pointers to neurons of previous layer
        private:
        float bias;
    };

    class AgregationNeuron: public Neuron
    {
    friend class ConclusionNeuron;
    public:
        AgregationNeuron(MuNeuron * prevLayer[], int count, float BIAS=0.001);
        ~AgregationNeuron();
        virtual void setState();
        MuNeuron ** Links; //array of pointers to neurons of previous layer
        int links_count;
    private:
        float bias;
    };

    class ConclusionNeuron: public Neuron
    {
    public:
        ConclusionNeuron(AgregationNeuron * prevLayer, float k_init[]);
        ~ConclusionNeuron();
        virtual void setState();
        float * k;
        int links_count;
    protected:
        InputNeuron ** Links; //array of pointers to neurons of previous layer
        AgregationNeuron * agrLink; //pointer to previous agregation layer
    };

    class ConclusionSumNeuron: public Neuron
    {
    public:
        ConclusionSumNeuron(ConclusionNeuron * prevLayer[], int count);
        ~ConclusionSumNeuron();
        virtual void setState();

    private:
        int links_count;
        ConclusionNeuron ** Links;
    };
    class AgregationSumNeuron: public Neuron
    {
        public:
        AgregationSumNeuron(AgregationNeuron * prevLayer[], int count);
        ~AgregationSumNeuron();
        virtual void setState();

    private:
        int links_count;
        AgregationNeuron ** Links;
    };

    class OutputNeuron: public Neuron
    {
    public:
        OutputNeuron(AgregationSumNeuron *  a, ConclusionSumNeuron *c);

        virtual void setState();

    private:
    AgregationSumNeuron * ASLink;
    ConclusionSumNeuron * CSLink;
    };

    class Layer
    {
    public:
    int neuron_count;
    };

    class Layer1: public Layer
    {
    public:
        Layer1(int x);
        ~Layer1();
        void inputSignal(float x[]);
    InputNeuron ** neurons;
    };

    class Layer2: public Layer
    {
    public:
        Layer2(Layer1 * input_layer);
        ~Layer2();
        void add(int index_x[], float b[], float c[], float sigma[], int count);
        void activation();
        MuNeuron ** neurons;
        Layer1 * l1;
    };

    class Layer3: public Layer
    {
    public:
        Layer3(Layer2 * input_layer, int count);
        ~Layer3();
        void add(int index_x[], int count);
        void activation();

        AgregationNeuron ** neurons;
        Layer2 * l2;
    private:
        int index;
    };

    class Layer4: public Layer
    {
    public:
        Layer4(Layer3 * input_layer);
        ~Layer4();
        void add(float k_input[]);
        void activation();
        ConclusionNeuron ** neurons;
        Layer3 * l3;
    private:
        int index;
    };

    class Layer5: public Layer
    {
    public:
        Layer5(Layer3* L3, Layer4 * L4);

        void activation(int neurons_to_activate=2);
        ConclusionSumNeuron * csneuron;
        AgregationSumNeuron * asneuron;
    private:
        Layer4 * l4;
        Layer3 * l3;

    };
    class Layer6: public Layer
    {
    public:
        Layer6(Layer5* L5);
        float activation();
        OutputNeuron * output;
    private:
        Layer5 * l5;
    };

    class NeuroNet
    {
    private:
        const int FSM[17][13] = //finite-state machine
        {
        {1, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
        {15, 15, 15, 15, 2, 15, 15, 15, 15, 15, 15, 15, 15},
        {15, 3, 15, 15, 15, 15, 15, 15, 15,  15, 16, 15, 15},
        {15, 15, 15, 15, 4, 15, 15, 15, 15, 15,  15, 15, 15},
        {15, 15, 15, 15, 15, 15, 5, 15, 15, 15,  15, 15, 15},
        {15, 15, 15, 15, 15, 15, 15, 6, 15, 15,  15, 15, 15},
        {15, 15, 15, 15, 15, 15, 15, 15, 7, 15, 15, 15, 15},
        {15,  15, 15, 15, 15, 8, 15, 15, 15, 15,  15, 15, 15},
        {15, 15, 9, 15, 15, 15, 5, 15, 15, 15,  15, 15, 15},
        {15, 15, 15, 15, 10, 15, 15, 15, 15, 15, 15, 15, 15},
        {15, 15, 15, 12, 15, 11, 15, 15,  15, 15, 15,  15, 15},
        {15, 15, 15, 12, 15, 11, 15, 15, 15, 15, 15, 15, 15},
        {15, 15, 15, 15, 15, 15, 15, 15, 15, 13, 15, 15, 15},
        {15, 15, 15, 15, 15, 15, 15, 15, 15, 13, 15, 15, 14},
        {14,  14, 14, 14, 14, 14, 14, 14, 14, 14,  14, 14, 14},
        {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15},
        {15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 16, 15, 15},
        };
    public:
        void Neuronet();
        char * viewneuroPath();
        int setneuroPath(char * x);
        char * viewtrainPath();
        int settrainPath(char * x);
        int restore();

        int backup();
        int train(float minimum_error, float ETA);

        float test(float vector[]);
        int test(float vector[], int n);
    //accessors:
        int inputs();
        float * get_limits(int i);
        int get_params_count(int index);
        float * get_params(int index, bool nostate);

        int rules();

        float * get_alpha();
        float * get_f();

    private:
        Layer1 * L1; //layer of input neurons x
        Layer2 * L2; //parametric layer of activation functions mu(x)
        Layer3 * L3; //agregation layer alpha=min{mu(x)}
        Layer4 * L4; //parametric layer of conclusion neurons z = alpha*f(x)
        Layer5 * L5; //sum layer: sum(z), sum(alpha)
        Layer6 * L6; //output layer z* = sum(z)/sum(alpha)

        char * path; //path to neuronet
        char * trainpath; //path to learning set
        float min_e; //minimum error of the net
        float eta; //teaching speed
        int column_index(char * s);

    };
}


#endif // TSKNETWORK_H

#include <tsknetwork.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <string.h>
using namespace std;

namespace TSKnetwork {

int extract_float(char * s, float * k, int len)
{
int j=0, v=0;
bool minus=false, fl=false;

while(s[j]!='\0')
{
    k[v] = 0;
    if(s[j]=='-')
    {
    minus=true;
    j++;
    }
    while((s[j]>='0')&&(s[j]<='9'))
    {
        k[v]= k[v]*10 + (s[j]-'0');
        j++;
        fl=true;
    }

    if(s[j]=='.')
    {
        j++;
        int af1=10;
        while((s[j]>='0')&&(s[j]<='9'))
        {
            k[v]= k[v] + (float)(s[j]-'0')/af1;
            af1*=10;
            j++;
        }
    }
    j++;

    if(fl)
    {
        fl=false;
        if(minus)
        {
            k[v]*=-1;
            minus=false;
        }
        v++;
        if(v>=len)
            return v;
    }

}
return v;
}
int extract_int(char *s, int * a, int len, int decr)
{
int j=0, v=0, fl=0;
for(int i=0; i<len;i++)
a[i] = 0;
    while(s[j]!='\0')
    {
        fl=j;
        while((s[j]>='0')&&(s[j]<='9'))
        {
            a[v]= a[v]*10 + (s[j]-'0');
            j++;
        }
        if(j!=fl)
        {
            a[v]-=decr;
            v++;
            if(v==len)
                return v;
        }
        j++;
    }
return v;
}
void cout_matrix(float ** A, int m, int n)
{
    FILE *fff;
    char * p = "D:\\r.txt";
        if((fff=fopen(p, "a+")) == NULL)
             return;
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<m; j++)
            fprintf(fff, "%.2f\t", A[i][j]);
        fprintf(fff, "\n");
    }
    fprintf(fff, "\n-------------\n\n");
    fclose(fff);
}
void solve(float ** T, int m, int n, int * indexes)
{
    int cur_pos = 0, cur=0;
    cout_matrix(T, m, n);
    for(int j=0; j<m-1; j++)
    {
        cur_pos=0;
        while(cur_pos<n)
        {
            int y=0;
            while(y<cur)
            {
                if(indexes[y]==cur_pos)
                    break;
                y++;
            }
            if((y!=cur)||(T[cur_pos][j]==0))
            {
                cur_pos++;
                continue;
            }
            float now = T[cur_pos][j];

            for(int l=0; l<n; l++)
                for(int t=j+1; t<m; t++)
                    if(l!=cur_pos)
                    T[l][t] = (T[l][t]*T[cur_pos][j]- T[l][j]*T[cur_pos][t])/T[cur_pos][j];
                for(int r=j; r<m; r++)
                T[cur_pos][r] = T[cur_pos][r]/now;
                indexes[cur] = cur_pos;
                cur++;
                for(int y=0; y<n;y++)
                    T[y][j] = 0;
                T[cur_pos][j]=1;
                break;
        }
        if(cur_pos==n)
        {
            indexes[cur] = -1;
            cur++;
        }
    }
    cout_matrix(T, m, n);
}


float Neuron::getState()
{
    return State;
}


 InputNeuron::InputNeuron(float input_x, float min_domain, float max_domain)
{
    a=min_domain;
    b=max_domain;
    setState(input_x);
}

void InputNeuron::setState(float input_x)
{
    if((input_x>=a)&&(input_x<=b))
    State=input_x;
    else
    {
        if(input_x<a)
            State=a;
        if(input_x>b)
            State=b;
    }
}


    MuNeuron::MuNeuron(InputNeuron * input, float B, float C, float SIGMA, float BIAS)
    {
        Links=input;
        b=B;
        c=C;
        sigma=SIGMA;
        bias = BIAS;
        setState();
    }
    void MuNeuron::setState()
    {
        float x = Links->getState();
        if(sigma==0)
            State=0;
        else
            State = 1/(exp(pow((x-c)/sigma, 2*b)));
        if(State<bias)
            State=0;

    }


    AgregationNeuron::AgregationNeuron(MuNeuron * prevLayer[], int count, float BIAS)
    {
        links_count=count;
        bias = BIAS;
        Links = new MuNeuron*[links_count];
        for(int i=0; i<count;i++)
            Links[i]=prevLayer[i];
        setState();
    }
    AgregationNeuron::~AgregationNeuron()
    {
        delete [] Links;
    }

    void AgregationNeuron::setState()
    {
        State=1;
        for(int i=0; i<links_count; i++)
            State*=Links[i]->getState();
        if(State<bias)
            State=0;
    }


    ConclusionNeuron::ConclusionNeuron(AgregationNeuron * prevLayer, float k_init[])
    {
        links_count = prevLayer->links_count;
        Links = new InputNeuron*[links_count];
        k = new float[links_count+1];
        agrLink = prevLayer;
        int i=0;
        for(i=0; i<links_count;i++)
        {
            Links[i]=prevLayer->Links[i]->Links;
            k[i]=k_init[i];
        }
        k[i] = k_init[i];
        setState();

    }
    ConclusionNeuron::~ConclusionNeuron()
    {
    delete [] Links;
    }

    void ConclusionNeuron::setState()
    {

        float kx;
        State=0;
        int i;
        for(i=0; i<links_count;i++)
        {
            kx = k[i]*Links[i]->getState();
            State+=kx;
        }
        State+=k[i];
        State *= agrLink->getState();
    }

    ConclusionSumNeuron:: ConclusionSumNeuron(ConclusionNeuron * prevLayer[], int count)
    {
        links_count = count;
        Links = new ConclusionNeuron*[links_count];
        for(int i=0; i<count;i++)
            Links[i]=prevLayer[i];
        setState();

    }
    ConclusionSumNeuron::~ConclusionSumNeuron()
    {
        delete [] Links;
    }
    void ConclusionSumNeuron::setState()
    {
        State=0;
        for(int i=0;i<links_count;i++)
            State+=Links[i]->getState();
    }

    AgregationSumNeuron::AgregationSumNeuron(AgregationNeuron * prevLayer[], int count)
    {
        links_count = count;

        Links = new AgregationNeuron*[links_count];
        for(int i=0; i<count;i++)
            Links[i]=prevLayer[i];
        setState();

    }
    AgregationSumNeuron::~AgregationSumNeuron()
    {
        delete [] Links;
    }
    void AgregationSumNeuron::setState()
    {
        State=0;
        for(int i=0;i<links_count;i++)
            State+=Links[i]->getState();
    }

    OutputNeuron::OutputNeuron(AgregationSumNeuron *  a, ConclusionSumNeuron *c)
    {
        ASLink = a;
        CSLink = c;
        setState();
    }

    void OutputNeuron::setState()
    {
        float AlphaSum = ASLink->getState();
        if(AlphaSum==0)
            AlphaSum=0.0001;
        State = CSLink->getState()/AlphaSum;
        if(State<0)
            State=0;
    }

    Layer1::Layer1(int x)
    {
    neuron_count = x;
    neurons = new InputNeuron*[x];
    for(int i=0; i<neuron_count; i++)
        neurons[i] = new InputNeuron();
    }
    Layer1::~Layer1()
    {
    delete [] neurons;
    }
    void Layer1::inputSignal(float x[])
    {
        int i=0;
        while(i<neuron_count)
        {
            neurons[i]->setState(x[i]);
            i++;
        }
    }


    Layer2::Layer2(Layer1 * input_layer)
    {
        l1 = input_layer;
    }
    Layer2::~Layer2()
    {
        for(int i=0;i<neuron_count;i++)
            delete [] neurons[i];
    delete [] neurons;
    }
    void Layer2::add(int index_x[], float b[], float c[], float sigma[], int count)
    {
        neuron_count = count;
        neurons = new MuNeuron*[neuron_count];
        for(int i=0;i<neuron_count;i++)
            neurons[i] = new MuNeuron(l1->neurons[index_x[i]], b[i], c[i], sigma[i]);
    }
    void Layer2::activation()
    {
        for(int i=0; i<neuron_count; i++)
            neurons[i]->setState();
    }


    Layer3::Layer3(Layer2 * input_layer, int count)
    {
        l2 = input_layer;
        neuron_count = count;
        neurons = new AgregationNeuron*[neuron_count];
        index=0;
    }
    Layer3::~Layer3()
    {
        for(int i=0;i<neuron_count;i++)
            delete [] neurons[i];
        delete [] neurons;
    }
    void Layer3::add(int index_x[], int count)
    {
            MuNeuron ** prevLayer = new MuNeuron*[count];
            for(int i=0;i<count;i++)
                prevLayer[i] = l2->neurons[index_x[i]];
            neurons[index] = new AgregationNeuron(prevLayer, count);
        if(index<neuron_count-1)
            index++;
    }
    void Layer3::activation()
    {
        for(int i=0; i<neuron_count; ++i)
            neurons[i]->setState();
    }

    Layer4::Layer4(Layer3 * input_layer)
    {
        l3 = input_layer;
        neuron_count = l3->neuron_count;
        neurons = new ConclusionNeuron*[neuron_count];
        index=0;
    }
    Layer4::~Layer4()
    {
        for(int i=0;i<neuron_count;i++)
            delete [] neurons[i];
        delete [] neurons;
    }
    void Layer4::add(float k_input[])
    {
        neurons[index] = new ConclusionNeuron(l3->neurons[index], k_input);
        if(index<neuron_count-1)
            index++;
    }

    void Layer4::activation()
    {
        for(int i=0; i<neuron_count; i++)
            neurons[i]->setState();
    }



    Layer5::Layer5(Layer3* L3, Layer4 * L4)
    {
        l3 = L3;
        l4 = L4;
        csneuron = new ConclusionSumNeuron(l4->neurons, l4->neuron_count);
        asneuron = new AgregationSumNeuron(l3->neurons, l3->neuron_count);
    }

    void Layer5::activation(int neurons_to_activate)
    {
        switch(neurons_to_activate)
        {
        case 0:
            asneuron->setState();
            break;
        case 1:
            csneuron->setState();
            break;
        case 2:
            csneuron->setState();
            asneuron->setState();
            break;
        }
    }


    Layer6::Layer6(Layer5* L5)
    {
        l5 = L5;
        output = new OutputNeuron(l5->asneuron, l5->csneuron);
    }

        float Layer6::activation()
    {
        output->setState();
        return output->getState();
    }



    void NeuroNet::Neuronet()
    {
        path = NULL;
        trainpath = NULL;
    }
    char * NeuroNet::viewneuroPath()
    {
        return path;
    }
    int NeuroNet::setneuroPath(char * x)
    {
        path = new char[strlen(x)+1];
        strcpy(path, x);
        return 0;
    }
    char * NeuroNet::viewtrainPath()
    {
        return trainpath;
    }
    int NeuroNet::settrainPath(char * x)
    {
        trainpath = new char[strlen(x)+1];
        strcpy(trainpath, x);
        return 0;
    }
    int NeuroNet::restore()
    {
        if(path == NULL)
            return -1;
        FILE * f=fopen(path, "r");
        if(f==NULL)
            return -1;
        char s[128];
        int x_count, domain_count=0;
        float AB[2];
        int mu_count, actual_mu_count=0;
        bool mu_param[3]={0, 0, 0};
        bool def_rule_count = false;
        bool def_k=false;
        int actual_k_count=0;
        int rule_count, actual_rule_count;
        int state=0;
        int links_i[20];

        float * b = new float[1];
        float * c= new float[1];
        float * sigma= new float[1];
        int * index= new int[1];

        while((fgets(s, 127, f))&&(state!=15))
        {
            if((s[0]=='#')||(s[0]=='\n'))
                continue;
            state = FSM[state][column_index(s)];
            switch(state)
            {
            case 0:
                break;
            case 1:
                break;
            case 2:
                if(sscanf(s, "%d", &x_count)>0)
                    L1 = new Layer1(x_count);
                else
                    state=15;
                break;
            case 3:
                L2 = new Layer2(L1);

                break;
            case 4:
                if(sscanf(s, "%d", &mu_count)>0)
                    {
                    delete [] b;
                    delete [] c;
                    delete [] sigma;
                    delete [] index;
                    b = new float[mu_count];
                    c = new float[mu_count];
                    sigma = new float[mu_count];
                    index = new int[mu_count];
                    actual_mu_count = 0;
                    break;
                    }
                else
                    state=15;
                break;
            case 5: //b
                if((actual_mu_count!=mu_count)&&(sscanf(s, "%f", &b[actual_mu_count])>0)&&(!mu_param[0])&&(!mu_param[1])&&(!mu_param[2]))
                        mu_param[0] = true;
                    else
                        state=15;
                break;
            case 6: //c
                if(actual_mu_count!=mu_count)
                if(sscanf(s, "%f", &c[actual_mu_count])>0)
                    if((mu_param[0])&&(!mu_param[1])&&(!mu_param[2]))
                        mu_param[1] = true;
                    else
                        state=15;
                break;
            case 7: //sigma
                if(actual_mu_count!=mu_count)
                if(sscanf(s, "%f", &sigma[actual_mu_count])>0)
                    if((mu_param[0])&&(mu_param[1])&&(!mu_param[2]))
                        mu_param[2] = true;
                    else
                        state=15;
                break;
            case 8: //link
                if(actual_mu_count!=mu_count)
                if(sscanf(s, "%i", &index[actual_mu_count])>0)
                    if((mu_param[0])&&(mu_param[1])&&(mu_param[2]))
                    {
                        if((index[actual_mu_count]>0)&&(index[actual_mu_count]<=x_count))
                            index[actual_mu_count]--;
                        else
                        {
                            state=15;
                            break;
                        }
                        actual_mu_count++;
                        if(actual_mu_count==mu_count)
                            {
                                for(int i=0;i<mu_count;i++)
                                L2->add(index, b, c, sigma, mu_count);
                            }
                        else
                        for(int i=0;i<3; i++)
                            mu_param[i]=0;
                    }
                    else
                        state=15;

                break;
            case 16:
                if(domain_count<=x_count)
                {
                    if(extract_float(s, AB, 2)==2)
                    {
                        L1->neurons[domain_count]->a = AB[0];
                        L1->neurons[domain_count]->b = AB[1];
                        domain_count++;
                    }
                    else
                        domain_count=x_count+1;
                }
                else
                    state=15;
                break;
            case 10: //3 layer neuron count
                if(sscanf(s, "%d", &rule_count)>0)
                    L3 = new Layer3(L2, rule_count);
                else
                    if(!strncmp(s, "default", strlen("default")))
                    {
                        rule_count = pow(mu_count/x_count, x_count);
                        L3 = new Layer3(L2, rule_count);

                        if((x_count<=0)||(mu_count<=0)||(mu_count%x_count!=0))
                        {
                            state = 15;
                            break;
                        }
                        int n = rule_count;
                        int m = x_count;
                        int ** A = new int *[n];
                        for(int i=0; i<n; i++)
                            A[i] = new int[m];

                        int k=1;
                        for(int j=0; j<m; j++)
                        {
                            k=pow(mu_count/x_count, m-j-1); //3^4, 3^3, ..3^0
                            int v=0;
                            int iter=0;
                            while(v<n)
                            {
                                for(int i=0; i<k; i++)
                                {
                                A[v][j] = j*mu_count/x_count + iter;
                                v++;
                                }
                                if(iter<mu_count/x_count-1)
                                iter++;
                                else
                                    iter=0;
                            }
                        }
                        for(int i=0; i<n; i++)
                        {
                            L3->add(A[i], x_count);
                            delete [] A[i];
                        }
                        delete []A;

                        def_rule_count=true;
                    }
                    else
                        state = 15;
                break;
            case 11: //links of 3rd layer
                if(!def_rule_count)
                {
                if(strchr(s, '#')!=NULL)
                    s[strchr(s, '#')-s]='\0';

                int links_i[20];
            //Adding new neuron links to Layer3:
                int v = extract_int(s, links_i, 20, 1);
                if(v)
                L3->add(links_i, v);
                }
                else
                    state=15;
                break;
            case 12:
                L4 = new Layer4(L3);
                break;
            case 13:
                float *  k = new float[x_count+1];
                if(strchr(s, '#')!=NULL)
                    s[strchr(s, '#')-s]='\0';
                if(!strncmp(s, "default", strlen("default")))
                {
                    if(!def_k) def_k=true;
                    else
                    {
                        state=15;
                        delete [] k;
                        break;
                    }

                    int g=0;
                    for(g=0;g<x_count;g++)
                        k[g]=1;
                    k[g]=0;
                    for(g=0; g<rule_count; g++)
                        L4->add(k);

                    actual_k_count=rule_count;
                }
                else
                {
                    if(actual_k_count>=rule_count)
                        {
                            state=15;
                            delete [] k;
                            break;
                        }
                if(extract_float(s, k, x_count+1))
                {
                    L4->add(k);
                    actual_k_count++;
                }
                }
                delete [] k;
                break;
            }
        }
        if((state!=15)&&(state!=0))
        {
        L5 = new Layer5(L3, L4);
        L6 = new Layer6(L5);
        }
    fclose(f);
    return !(state==15);
    }
    int NeuroNet::backup()
    {
        if(path==NULL)
            return -1;
        char s[128];
        FILE *f;
        if((f=fopen(path, "w")) == NULL)
             return -1;

        fprintf(f, "#Configuration file for TSK Neuronet.\n[1 Layer]\n\n");
        fprintf(f, "Neuron Count =  %d\n", L1->neuron_count);
        for(int i=0; i<L1->neuron_count; i++)
        fprintf(f, "domain =  %.3f, %.3f\n", L1->neurons[i]->a, L1->neurons[i]->b);
        fprintf(f, "\n[2 Layer]\n\n");
        fprintf(f, "Neuron Count =  %d\n", L2->neuron_count);
        for(int i=0; i<L2->neuron_count; i++)
        {
            int j=0;
            for(j=0; j<L1->neuron_count; j++)
                if(L2->neurons[i]->Links == L1->neurons[j])
                    break;

        fprintf(f, "b = %.2f\nc = %.2f\nsigma = %.2f\nlink = %d\n\n", L2->neurons[i]->b, L2->neurons[i]->c, L2->neurons[i]->sigma, j+1);
        }
        fprintf(f, "[3 Layer]\n\n");
        fprintf(f, "Neuron Count =  %d\n", L3->neuron_count);
        for(int i=0; i<L3->neuron_count; i++)
        {
            fprintf(f, "link = ");
            bool first=true;
            for(int k=0; k<L3->neurons[i]->links_count; k++)
            for(int j=0; j<L2->neuron_count; j++)
            {
                if(L3->neurons[i]->Links[k]==L2->neurons[j])
                if(first)
                {
                    first=false;
                    fprintf(f, "%d", j+1);
                }
                else
                    fprintf(f, ", %d", j+1);

            }
            fprintf(f, "\n");
        }
        fprintf(f, "\n\n[4 Layer]\n\n");
        for(int i=0; i<L4->neuron_count;i++)
        {
            fprintf(f, "k = %.3f", L4->neurons[i]->k[0]);
            for(int j=1; j<=L4->neurons[i]->links_count; j++)
                fprintf(f, ", %.3f", L4->neurons[i]->k[j]);
            fprintf(f, "\n");
        }
        fprintf(f, "\n");

        fclose(f);
        return 0;

    }
    int NeuroNet::train(float minimum_error, float ETA)
    {
        if(trainpath==NULL)
            return -1;
        char s[128];
        min_e = minimum_error;
        eta = ETA;
        FILE * f=fopen(trainpath, "r");
        if(f==NULL)
            return -1;
        int N = L1->neuron_count;
        int M = L3->neuron_count;

        if(fgets(s, 127, f)==NULL)
            return -1;
        int L;
        int iteration_count=0;
        float error=min_e+1;  //squared error measure (the discrepancy or error)
        if(sscanf(s, "%d", &L)<=0)
            return -1;
        float * e = new float[L]; //teacher output
        float ** x = new float *[L];
        float ** Ae = new float*[L];

        for(int a=0; a<L; a++)
        {
            if(fgets(s, 127, f)==NULL)
            {
                if(a==0)
                return -1;
                L=a;
                break;
            }
            x[a] = new float[N];
            extract_float(s, x[a], N);
            if(fgets(s, 127, f)==NULL)
            return -1;
            if(sscanf(s, "%f", &e[a])<=0)
            return -1;
            Ae[a] = new float[(N+1)*M + 1];
        }
        fclose(f);

        //adjusting k
        //if(L<(N+1)*M)
        //	return -1;
        int index;
        while((error>min_e)&&(iteration_count<1000))
        {
        error=0;
        iteration_count++;

        for(int i=0; i<L; i++)
        {
            L1->inputSignal(x[i]);
            L2->activation();
            L3->activation();
            L5->activation(0);
            index=0;
            //alpha', x
            float l5;
            for(int j=0; j<M; j++)
            {
                for(int r=0; r<(N+1); r++)
                {
                    l5 = L5->asneuron->getState();
                    if(l5==0) l5=0.0001;
                    Ae[i][index] = L3->neurons[j]->getState();
                    Ae[i][index]=Ae[i][index]/l5;
                    if(r!=0)
                        Ae[i][index]*=x[i][r-1];
                    index++;
                }
            }
            Ae[i][(N+1)*M] = e[i];
        }
        int * ind = new int[(N+1)*M];
        solve(Ae, (N+1)*M+1, L, ind);
        index=0;

        for(int i=0; i<M; i++)
        {
            (ind[index]==-1)?L4->neurons[i]->k[L4->neurons[i]->links_count] =0:
            L4->neurons[i]->k[L4->neurons[i]->links_count] = Ae[ind[index]][(N+1)*M];
            index++;
            for(int j=0; j<N; j++)
            {
                (ind[index]==-1)?L4->neurons[i]->k[j] =0:
                L4->neurons[i]->k[j] = Ae[ind[index]][(N+1)*M];
                index++;
            }

        }
        float * epsilon = new float[L]; //neural net output
        for(int i=0; i<L; i++)
        {
            float z = test(x[i]);
            epsilon[i] = z - e[i];
            error+=epsilon[i]*epsilon[i];
        }
        error*=0.5;
        float f_sum, grad_c, grad_sigma, grad_b;
        for(int t=0; t<4; t++)
        for(int j=0; j<L; j++)
        {
                L1->inputSignal(x[j]);
                L2->activation();
                L3->activation();
                L4->activation();
                L5->activation(1);
                f_sum = L5->csneuron->getState()*eta*epsilon[j];
                for(int i=0; i<L2->neuron_count; i++)
                {
                    grad_c = 2*(L2->neurons[i]->Links->getState() - L2->neurons[i]->c)/(pow(L2->neurons[i]->sigma, 2));
                    grad_c*=L2->neurons[i]->getState();
                    grad_sigma = 2*(L2->neurons[i]->Links->getState() - L2->neurons[i]->c)/(pow(L2->neurons[i]->sigma, 3));
                    grad_sigma*=L2->neurons[i]->getState();

                    L2->neurons[i]->c -=f_sum*grad_c;
                    L2->neurons[i]->sigma -=f_sum*grad_sigma;
                }
        }

        }
        for(int i=0; i<L;i++)
        {
            delete [] x[i];
            delete [] Ae[i];
        }
        delete [] x;
        delete [] Ae;
        return 0;
    }
    float NeuroNet::test(float vector[])
    {
        L1->inputSignal(vector);
        L2->activation();
        L3->activation();
        L4->activation();
        L5->activation();
        return L6->activation();
    }
    int NeuroNet::test(float vector[], int n)
    {
        L1->inputSignal(vector);
        L2->activation();
        L3->activation();
        L4->activation();
        L5->activation();
        float z = L6->activation();
        int Z;
        switch(n)
        {
        case 0:
            Z = ceil(z);
            break;
        case 1:
            Z = floor(z);
            break;
        default:
            Z = (int)z;
        }
        return Z;
    }
//accessors:
    int NeuroNet::inputs()
    {
        if(L1!=NULL)
        return L1->neuron_count;
        return -1;
    }
    float * NeuroNet::get_limits(int i)
    {
        if(L1==NULL)
            return NULL;
        if(i>=L1->neuron_count)
            return NULL;
        float * param = new float[2];
        param[0] = L1->neurons[i]->a;
        param[1] = L1->neurons[i]->b;
        return param;
    }
    int NeuroNet::get_params_count(int index)
    {
        if(index>=L1->neuron_count)
            return -1;
        int n=0;
        for(int i=0; i<L2->neuron_count; i++)
            if(L2->neurons[i]->Links == L1->neurons[index])
                n++;
        if(n==0)
            return -1;
        return n;
    }
    float * NeuroNet::get_params(int index, bool nostate)
    {
        int n=get_params_count(index);
        if(n==-1)
            return NULL;
        float * param;
        int j=0;
        if(!nostate)
        {
        param = new float[n*3];
        for(int i=0; i<L2->neuron_count; i++)
        {
            if(L2->neurons[i]->Links == L1->neurons[index])
            {
                param[j++]=L2->neurons[i]->b;
                param[j++] = L2->neurons[i]->c;
                param[j++] = L2->neurons[i]->sigma;
            }
        }
        }
        else
        {
            param = new float[n];
            j=0;
            for(int i=0; i<L2->neuron_count; i++)
            {
                if(L2->neurons[i]->Links == L1->neurons[index])
                {
                    param[j++] = L2->neurons[i]->getState();
                }
            }

        }
        return param;
    }
    int NeuroNet::rules()
    {
        return L3->neuron_count;
    }

    float * NeuroNet::get_alpha()
    {
        float * alphas = new float[L3->neuron_count];
        for(int i=0; i<L3->neuron_count; i++)
        {
            alphas[i] = L3->neurons[i]->getState();
        }
        return alphas;
    }
    float * NeuroNet::get_f()
    {
        float * cons = new float[L4->neuron_count];
        for(int i=0; i<L4->neuron_count; i++)
        {
            cons[i] = L4->neurons[i]->getState();
        }
        return cons;
    }


    int NeuroNet::column_index(char * s)
    {
        if(!strncmp(s, "[1 Layer]", strlen("[1 Layer]")))
            return 0;
        if(!strncmp(s, "[2 Layer]", strlen("[2 Layer]")))
            return 1;
        if(!strncmp(s, "[3 Layer]", strlen("[3 Layer]")))
            return 2;
        if(!strncmp(s, "[4 Layer]", strlen("[4 Layer]")))
            return 3;
        if(!strncmp(s, "Neuron Count = ", strlen("Neuron Count = ")))
        {
            strcpy(s, &s[strlen("Neuron Count = ")]);
            return 4;
        }
        if(!strncmp(s, "link = ", strlen("link = ")))
        {
            strcpy(s, &s[strlen("link = ")]);
            return 5;
        }
        if(!strncmp(s, "b = ", strlen("b = ")))
        {
            strcpy(s, &s[strlen("b = ")]);
            return 6;
        }
        if(!strncmp(s, "c = ", strlen("c = ")))
        {
            strcpy(s, &s[strlen("c = ")]);
            return 7;
        }
        if(!strncmp(s, "sigma = ", strlen("sigma = ")))
        {
            strcpy(s, &s[strlen("sigma = ")]);
            return 8;
        }
        if(!strncmp(s, "k = ", strlen("k = ")))
        {
            strcpy(s, &s[strlen("k = ")]);
            return 9;
        }
        if(!strncmp(s, "domain = ", strlen("domain = ")))
        {
            strcpy(s, &s[strlen("domain = ")]);
            return 10;
        }

        return 11;
    }
}


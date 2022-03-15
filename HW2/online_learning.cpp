#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

typedef struct _binomial_info {
    double alpha;
    double beta;
    double p;
} binomial_info;

double factorial(double n) {
    if (n <= 1.0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

double count_likelihood(double p, double alpha, double beta) {
    return factorial(alpha + beta) / (factorial(alpha) * factorial(beta)) * pow(0.5, alpha) * pow(0.5, beta);
}

void online_learning(binomial_info& info, string outcome) {
    unsigned long cnt_0 = 0, cnt_1;
    for (unsigned long i = 0; i < outcome.size(); i++) 
        if (outcome[i] == '1')
            cnt_0++;
    cnt_1 = outcome.size() - cnt_0;
    info.alpha += cnt_0;
    info.beta += cnt_1;
    info.p = count_likelihood(info.p, info.alpha, info.beta);
}

int main(int argc, char * argv[]) {
    string fname("testfile.txt");
    ifstream fin(fname, ios::in);
    binomial_info info;

    unsigned int alpha, beta;
    unsigned int case_cnt = 0;
    string line;


    cin >> alpha >> beta;
    info.alpha = alpha;
    info.beta = beta;
    info.p = 0.5;
    while ((fin >> line)) {
        case_cnt++;
        alpha = info.alpha;
        beta = info.beta;
        online_learning(info, line);

        cout << "case " << case_cnt << ": " << line << endl;
        cout << "Likelihood: " << info.p << endl;
        cout << "Beta prior:  a = " << alpha << " b = " << beta << endl;
        cout << "Beta posterior:  a = " << info.alpha << " b = " << info.beta << endl;
        cout << endl;
    }

    return 0;
}
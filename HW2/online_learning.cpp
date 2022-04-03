#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;

typedef struct _binomial_info {
    double alpha;
    double beta;
    double likelihood;
} binomial_info;

double factorial(double n) {
    if (n <= 1.0) {
        return 1;
    } else {
        return n * factorial(n - 1);
    }
}

double count_likelihood(double alpha, double beta, double p) {
    return factorial(alpha + beta) / (factorial(alpha) * factorial(beta)) * pow(p, alpha) * pow(1 - p, beta);
}

void online_learning(binomial_info& info, string outcome) {
    double cnt_0, cnt_1 = 0;
    for (unsigned long i = 0; i < outcome.size(); i++) 
        if (outcome[i] == '1')
            cnt_1 += 1.0;
    cnt_0 = outcome.size() - cnt_1;
    info.likelihood = count_likelihood(cnt_1, cnt_0, cnt_1 / (cnt_0 + cnt_1));
    info.alpha += cnt_1;
    info.beta += cnt_0;
}

int main(int argc, char * argv[]) {
    string fname("testfile.txt");
    ifstream fin(fname, ios::in);
    binomial_info info;

    unsigned int alpha, beta;
    unsigned int case_cnt = 0;
    cout << "parameter a for the initial beta prior :"; 
    cin >> alpha ;
    cout << "parameter b for the initial beta prior :"; 
    cin >>  beta;

    info.alpha = alpha;
    info.beta = beta;
    string line;
    while ((fin >> line)) {
        case_cnt++;
        alpha = info.alpha;
        beta = info.beta;
        online_learning(info, line);

        cout << "case " << case_cnt << ": " << line << endl;
        cout << "Likelihood: " << info.likelihood << endl;
        cout << "Beta prior:  a = " << alpha << " b = " << beta << endl;
        cout << "Beta posterior:  a = " << info.alpha << " b = " << info.beta << endl;
        cout << endl;
    }

    return 0;
}
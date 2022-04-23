#include <iomanip>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
#include <unistd.h>
using namespace std;

#define MINIMUM 1e-5
#define CHANNEL 256
#define BIN_SIZE 8

typedef struct _datapair {
    unsigned char row;
    unsigned char col;
    unsigned char label;
    unsigned char *img;
} datapair;

unsigned byte2unsigned(char* carr) {
    unsigned ret = 0;
    for (int i = 0; i < 4; i++) {
        ret <<= 8;
        ret |= (unsigned char)carr[i];
    }
    return ret;
}

inline double gaussian(double val, double mean, double std) {
    return exp(-0.5 * pow((val - mean) / std, 2)) / (std * sqrt(2 * M_PI)); 
}

vector<datapair> read_datapairs(const string& img_file, const string& label_file) {
    vector<datapair> ret;
    datapair * tmp_data = NULL;

    ifstream fin_img(img_file, ios::binary);
    ifstream fin_label(label_file, ios::binary);
    char buffer[4], dummy[8];

    // magic number
    fin_img.read(buffer, 4);
    byte2unsigned(buffer);

    // length
    fin_img.read(buffer, 4);
    unsigned num_data = byte2unsigned(buffer);
    
    // row col
    fin_img.read(buffer, 4);
    unsigned row = byte2unsigned(buffer);
    fin_img.read(buffer, 4);
    unsigned col = byte2unsigned(buffer);

    // dummy of label
    fin_label.read(dummy, 8);

    char byte;
    unsigned num_pixels = row * col;
    for (unsigned i = 0; i < num_data; i++) {

        tmp_data = new datapair;
        if (!tmp_data) {
            cout << "malloc err in tmp_data" << endl;
            exit(0);
        }

        tmp_data->img = new unsigned char [num_pixels];
        if (tmp_data->img) {

            tmp_data->row = row;
            tmp_data->col = col;
            
            fin_label.read(&byte, 1);
            tmp_data->label = (unsigned char)byte;

            for (unsigned i = 0; i < num_pixels; i++) {
                fin_img.read(&byte, 1);
                tmp_data->img[i] = (unsigned char)byte / 128;
            }

            ret.push_back(*tmp_data);
        }  
    }

    fin_img.close();
    fin_label.close();

    return ret;
}

void display(datapair& dp, unsigned char pred) {
    cout << (unsigned int)pred << ":" << endl;
    for (int i = 0; i < dp.row; i++) {
        for (int j = 0; j < dp.col; j++) {
            cout << (unsigned int)dp.img[i * dp.row + j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

vector<double> init_lambda(vector<unsigned> &category_num, unsigned total) {
    vector<double> ret(category_num.size(), 0);
    for (unsigned long i = 0; i < category_num.size(); i++) {
        ret[i] = (double)(category_num[i]) / total;
    }
    return ret;
}

vector<vector<double> > init_p() {
    vector<vector<double> > p(10, vector<double>(784, 0.0));
    for (unsigned char j = 0; j < 10; j++) {
        for (unsigned bin = 0; bin < 784; bin++) {
            p[j][bin] = 0.4 + ((double) rand() / (RAND_MAX)) / 5; // range from [0.4 -> 0.6]
        }
    }
    return p;
}

void E_step(const vector<datapair> &train_data, const vector<vector<double> > &p, const vector<double> &lambda, vector<vector<double> > &w) {
    double w_dividend;
    for (unsigned long i = 0; i < train_data.size(); i++) {
        w_dividend = 0.0;
        for (unsigned char j = 0; j < 10; j++) {
            w[j][i] = lambda[j]; 
            for (unsigned bin = 0; bin < 784; bin++) {
                w[j][i] *= train_data[i].img[bin] ? p[j][bin] : (1 - p[j][bin]);
            }
            w_dividend += w[j][i];
        }
        for (unsigned char j = 0; j < 10; j++) {
            w[j][i] /= w_dividend;
        }
    }
}

void M_step(const vector<datapair> &train_data, const vector<vector<double> > &w, vector<vector<double> > &p, vector<double> &lambda) {
    double lambda_dividend, p_dividend;
    // count lambda
    for (unsigned char j = 0; j < 10; j++) {
        lambda_dividend = 0.0;
        for (unsigned long i = 0; i < train_data.size(); i++) {
            lambda_dividend += w[j][i];
        }
        lambda[j] = lambda_dividend / train_data.size();

        for (unsigned bin = 0; bin < 784; bin++) {
            p_dividend = 0.0;
            for (unsigned long i = 0; i < train_data.size(); i++) {
                p_dividend += (w[j][i] * train_data[i].img[bin]);
            }
            p[j][bin] = p_dividend / lambda_dividend;
        }
    }
}

void count_zw_map(const vector<datapair> &train_data, const vector<vector<double> > &w, vector<unsigned char> &zw_map) {
    vector<vector<unsigned>> zw_cnt(10, vector<unsigned>(10, 0));
    double max;
    unsigned long max_j;
    for (unsigned long i = 0; i < train_data.size(); i++) {
        max = -1;
        max_j = -1;
        for (unsigned long j = 0; j < 10; j++) {
            if (w[j][i] > max) {
                max = w[j][i];
                max_j = j;
            }
        }
        zw_cnt[train_data[i].label][max_j]++;
    }

    for (unsigned long i = 0; i < 10; i++) {
        max = -1;
        max_j = -1;
        for (unsigned long j = 0; j < 10; j++) {
            if (zw_cnt[i][j] > max) {
                max = zw_cnt[i][j];
                max_j = j;
            }
        }
        zw_map[i] = max_j;
    }
}

double inference(const vector<datapair> &train_data, const vector<unsigned char> &zw_map, const vector<vector<double> > &p, const vector<vector<double> > &p_cache) {
    unsigned char cur_w;
    double diff = 0.0;

    for (unsigned long i = 0; i < 10; i++) {
        for (unsigned bin = 0; bin < 784; bin++) {
            diff += abs(p[i][bin] - p_cache[i][bin]);
        }
    }

    for (unsigned char j = 0; j < 10; j++) {
        cout << "class " << (unsigned)j << ":" << endl;
        cur_w = j;
        for (unsigned bin = 0; bin < 784; bin++) {
            cout << (p[cur_w][bin] > 0.5 ? 1 : 0) << " ";
            if (bin % 28 == 27)
                cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    fflush(stdout);

    return diff;
}

void confusion_matrix(const vector<datapair> &train_data, const vector<unsigned char> &zw_map, const vector<vector<double> > &w) {
    vector<unsigned char> pred(train_data.size(), 0);
    unsigned tp, fn, fp, tn;
    double max;
    unsigned char max_j;
    for (unsigned long i = 0; i < train_data.size(); i++) {
        max = 0;
        for (unsigned long j = 0; j < 10; j++) {
            if (w[zw_map[j]][i] > max) {
                max = w[zw_map[j]][i];
                max_j = j;
            }
        }
        pred[i] = max_j;
    }

    for (unsigned long j = 0; j < 10; j++) {
        tp = fn = fp = tn = 0;
        for (unsigned long i = 0; i < train_data.size(); i++) {
            if (train_data[i].label == j) {
                if (pred[i] == j) {
                    tp++;
                } else {
                    fn++;
                }
            } else {
                if (pred[i] == j) {
                    fp++;
                } else {
                    tn++;
                }
            }
        }
        cout << "Confusion Matrix " << j << ":" << endl;
        cout << "Is number " << j << setw(13) << tp << setw(10) << fn << endl;
        cout << "Isn't number " << j << setw(10) << fp << setw(10) << tn << endl << endl;
        cout << "Sensitivity (Successfully predict number " << j << ")      : " << (double)tp / (tp + fn) << endl; 
        cout << "Sensitivity (Successfully predict not number " << j << ")  : " << (double)tn / (fp + tn) << endl; 
    }
}

vector<vector<double> > train_p_lambda(const vector<datapair> &train_data, vector<double> &lambda, vector<vector<double> > &p) {
    vector<vector<double>> w(10, vector<double>(train_data.size(), 1.0));
    vector<vector<double> > p_cache;
    vector<unsigned char> zw_map(10, -1);

    int cnt = 0, max_cnt = 100;
    double diff, eps = 10;
    while (cnt < max_cnt) {
        cnt ++;
        p_cache = p;
        E_step(train_data, p, lambda, w);
        M_step(train_data, w, p, lambda);
        count_zw_map(train_data, w, zw_map);
        diff = inference(train_data, zw_map, p, p_cache);
        cout << "No. of Iteration: " << cnt << ", Difference: " << diff << endl << endl;
        if (diff < eps) {
            break;
        }
    }
    confusion_matrix(train_data, zw_map, w);

    return w;
}

int main(int argc, char * argv[]) {
    const string fname_train_img = "train-images.idx3-ubyte";
    const string fname_train_label = "train-labels.idx1-ubyte";
    const string fname_test_img = "t10k-images.idx3-ubyte";
    const string fname_test_label = "t10k-labels.idx1-ubyte";

    vector<datapair> train_data =  read_datapairs(fname_train_img, fname_train_label);

    srand(getpid() ^ time(0));
    vector<double> lambda(10, 0.1);
    vector<vector<double> > p = init_p(); // for training
    vector<vector<double> > w = train_p_lambda(train_data, lambda, p);
    return 0;

}


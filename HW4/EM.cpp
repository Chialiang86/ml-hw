#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
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
    unsigned magic = byte2unsigned(buffer);

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

    cout << "magic num = " << magic << " num of data = " << num_data << " row = " << row << " col = " << col << endl;
    
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

vector<unsigned> count_category_num(vector<datapair> &dp_vec) {
    vector<unsigned> ret(10, 0);
    for (long unsigned i = 0; i < dp_vec.size(); i++)
        ret[dp_vec[i].label] ++;
    return ret;
}

// vector<unsigned> count_k(vector<datapair> &dp_vec) {
//     unsigned total = dp_vec[0].row * dp_vec[0].col;
//     vector<unsigned> k(dp_vec.size(), 0);

//     for (long unsigned i = 0; i < dp_vec.size(); i++) {
//         for (unsigned b = 0; b < total; b++) {
//             k[i] += dp_vec[i].img[b] / 128;
//         }
//     }
//     return k;
// }

vector<double> init_lambda(vector<unsigned> &category_num, unsigned total) {
    vector<double> ret(category_num.size(), 0);
    for (unsigned long i = 0; i < category_num.size(); i++) {
        ret[i] = (double)(category_num[i]) / total;
    }

    return ret;
}

vector<vector<double> > init_p() {
    vector<vector<double> > p(10, vector<double>(784, 0.5));
    return p;
}

void E_step(const vector<datapair> &train_data, const vector<vector<double> > &p, const vector<double> &lambda, vector<vector<double> > &w) {
    double sum;
    for (unsigned long i = 0; i < train_data.size(); i++) {
        sum = 0.0;
        for (unsigned char j = 0; j < 10; j++) {
            w[j][i] = lambda[j]; 
            for (unsigned bin = 0; bin < 784; bin++) {
                w[j][i] *= train_data[i].img[bin] ? p[j][bin] : (1 - p[j][bin]);
            }
            sum += w[j][i];
        }
        for (unsigned char j = 0; j < 10; j++) {
            w[j][i] /= sum;
            // cout << "w[" << (unsigned)j << "][" << i << "] = " << w[j][i] << endl;
        }
    }
}

void M_step(const vector<datapair> &train_data, const vector<vector<double> > &w, vector<vector<double> > &p, vector<double> &lambda) {
    double sum;
    vector<double> w_sum(10, 0.0);
    // count lambda
    for (unsigned char j = 0; j < 10; j++) {
        sum = 0.0;
        for (unsigned long i = 0; i < train_data.size(); i++) {
            sum += w[j][i];
        }
        w_sum[j] = sum;
        lambda[j] = sum / train_data.size();
        // cout << "l[" << (unsigned)j << "] = " << lambda[j] << " ";
    }
    cout << endl;

    // count p
    for (unsigned char j = 0; j < 10; j++) {
        for (unsigned bin = 0; bin < 784; bin++) {
            sum = 0.0;
            for (unsigned long i = 0; i < train_data.size(); i++) {
                sum += (w[j][i] * train_data[i].img[bin]);
            }
            p[j][bin] = sum / w_sum[j];
        }
    }
}

void inference(vector<vector<double> > &p) {
    for (unsigned char j = 0; j < 10; j++) {
        cout << "class " << (unsigned)j << ":" << endl;
        for (unsigned bin = 0; bin < 784; bin++) {
            cout << (p[j][bin] > 0.3 ? 1 : 0) << " ";
            if (bin % 28 == 27)
                cout << endl;
        }
        cout << endl;
    }
    cout << endl;
    fflush(stdout);
}

// double count_p(double lambda_j, double p_j, unsigned k_i, unsigned pixels=784) {
//     double exp_10 = 0, ret;
//     // log(C(pixels, k_i)) * p_j^k_i * (1 - p_j)^(pixels - k_i))

//     // log (C(pixels, k_i)) = log (pixels! / (pixels - k_i)!) - log (k_i!)
//     for (unsigned i = pixels - k_i + 1; i < pixels; i++) {
//         exp_10 += log10(i); // log (pixels! / (pixels - k_i)!)
//     }
//     for (unsigned i = 1; i < k_i; i++) {
//         exp_10 -= log10(i); // log (k_i!)
//     }

//     // log(p_j^k_i * (1 - p_j)^(pixels - k_i))
//     exp_10 += (k_i * log10(p_j));
//     exp_10 += ((pixels - k_i) * log10(1.0 - p_j));
//     exp_10 += log10(lambda_j);
//     ret = pow(10.0, exp_10);
//     return exp_10;
// }

vector<vector<double> > train_p_lambda(const vector<datapair> &train_data, vector<double> &lambda, vector<vector<double> > &p) {
    vector<vector<double>> w(10, vector<double>(train_data.size(), 1.0));

    int cnt = 0;
    double w_sum, divident, divisor;
    while (cnt < 100) {
        // for (int j = 0; j < 10; j++) {
        //     cout << "lambda " << j << " = " << lambda[j] << endl;
        // }
        // for (int j = 0; j < 10; j++) {
        //     cout << "p " << j << " = " << p[j] << endl;
        // }
        // cout << endl;
        
        cnt ++;
        cout << "[iteration = " << cnt << "] E step running..." << endl;
        E_step(train_data, p, lambda, w);
        cout << "[iteration = " << cnt << "] M step running..." << endl;
        M_step(train_data, w, p, lambda);
        inference(p);

        // // E step
        // for (int ki = 0; ki < 785; ki++) {
        //     divisor = 0.0;
        //     for (int i = 0; i  < 10; i++) {
        //         w[i][ki] = count_p(lambda[i], p[i], ki);
        //         divisor += w[i][ki];
        //     }
        //     for (int i = 0; i < 10; i++) {
        //         w[i][ki] /= divisor;
        //         printf("w[%d][%d] = %f\n", i, ki, w[i][ki]);
        //     }
        // }
        
        // // M step
        // for (int i = 0; i < 10; i++) {
        //     w_sum = divident = divisor = 0;
        //     for (int id = 0; id < n; id++) {
        //         w_sum += w[i][k[id]];
        //         divident += w[i][k[id]] * k[id]; // sum(w_id_i * k_id) from id = 0 -> n
        //         divisor += w[i][k[id]] * 784;
        //     }
        //     lambda[i] = w_sum / n;
        //     p[i] = divident / divisor;
        //     // cout << "p[" << i << "] = " << p[i] << " ";
        // }
        // cout << endl;

    }

    return w;
}

// vector<unsigned char> inference(vector<vector<double>> &w, vector<unsigned> &k, vector<datapair> &test_data) {
//     unsigned long test_num = k.size();
//     vector<unsigned char> ret(test_num, -1);
//     double max;
    
//     for (unsigned long id = 0; id < test_num; id++) {
//         max = -1;
//         for (unsigned long i = 0; i < 10; i++) {
//             if (max < w[i][k[id]]) {
//                 max = w[i][k[id]];
//                 ret[id] = i;
//             }
//         }
//         cout << "predict : " << ret[id] << " answer : " << test_data[id].label << endl;
//     }
//     return ret;
// }

int main(int argc, char * argv[]) {
    const string fname_train_img = "train-images.idx3-ubyte";
    const string fname_train_label = "train-labels.idx1-ubyte";
    const string fname_test_img = "t10k-images.idx3-ubyte";
    const string fname_test_label = "t10k-labels.idx1-ubyte";

    vector<datapair> train_data =  read_datapairs(fname_train_img, fname_train_label);
    // vector<datapair> test_data =  read_datapairs(fname_test_img, fname_test_label);

    vector<unsigned> category_nums = count_category_num(train_data);
    
    // vector<unsigned> k_train = count_k(train_data);
    // vector<unsigned> k_test = count_k(train_data);
    vector<double> lambda = init_lambda(category_nums, train_data.size()); // for training
    vector<vector<double> > p = init_p(); // for training

    vector<vector<double> > w = train_p_lambda(train_data, lambda, p);
    // vector<unsigned char> res = inference(w, k_test, test_data);
    return 0;

}


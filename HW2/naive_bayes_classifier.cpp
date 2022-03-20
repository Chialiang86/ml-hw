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

vector<datapair> read_datapairs(const string img_file, const string label_file) {
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
                tmp_data->img[i] = (unsigned char)byte;
            }

            ret.push_back(*tmp_data);
        }  
    }

    fin_img.close();
    fin_label.close();

    return ret;
}

void display(datapair dp) {
    cout << (unsigned int)dp.label << ":" << endl;
    for (int i = 0; i < dp.row; i++) {
        for (int j = 0; j < dp.col; j++) {
            cout << ((unsigned int)dp.img[i * dp.row + j] < 128 ?  0 : 1) << " ";
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

vector<unsigned char> count_bins(datapair &dp, unsigned char bin_size=BIN_SIZE) {
    unsigned total = dp.row * dp.col;
    vector<unsigned char> bins(total, 0);
    for (unsigned i = 0; i < total; i++) {
        bins[i] = dp.img[i] / bin_size;
    }
    return bins;
}

void dump_model(vector<vector<vector<double> > >& train_discrete_model){
    for (long unsigned i = 0; i < train_discrete_model.size(); i++) {
        for (long unsigned j = 0; j < train_discrete_model[i].size(); j++) {
            for (long unsigned k = 0; k < train_discrete_model[i][j].size(); k++) {
                cout << train_discrete_model[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

vector<vector<vector<double> > > create_discrete_model(vector<datapair>& dp_vec, unsigned char bin_size=BIN_SIZE, double init_num=MINIMUM) {
    vector<vector<vector<double> > > ret(10, vector<vector<double> >(784, vector<double>((unsigned)(CHANNEL / bin_size), init_num)));

    for (long unsigned i = 0; i < dp_vec.size(); i++) {
        for (unsigned b = 0; b < 784; b++) {
            ret[dp_vec[i].label][b][dp_vec[i].img[b] / bin_size] += 1.0;
        }
    }

    return ret;
}

vector<vector<vector<double> > > create_continuous_model(vector<datapair>& dp_vec, double init_num=MINIMUM) {
    vector<vector<vector<double> > > ret(10, vector<vector<double> >(784, vector<double>(CHANNEL, 0.5)));

    for (long unsigned i = 0; i < dp_vec.size(); i++) {
        for (unsigned b = 0; b < 784; b++) {
            ret[dp_vec[i].label][b][dp_vec[i].img[b]] += 1;
        }
    }

    double mean, std, sum;
    for (long unsigned i = 0; i < ret.size(); i++) {
        for (long unsigned b = 0; b < ret[i].size(); b++) {
            mean = 0.0;
            sum = 0.0;
            for (long unsigned val = 0; val < ret[i][b].size(); val++) {
                mean += ret[i][b][val] * val;
                sum += ret[i][b][val];
            }
            mean /= sum;
            
            std = 0.0;
            for (long unsigned val = 0; val < ret[i][b].size(); val++) {
                std += ret[i][b][val] * pow(val - mean, 2) ;
            }
            std = sqrt(std / sum);

            for (long unsigned val = 0; val < ret[i][b].size(); val++) {
                ret[i][b][val] = gaussian(val, mean, std);
            }
        }
    }

    return ret;
}

vector<double> count_discrete_posterior(vector<vector<vector<double> > >& train_discrete_model, vector<unsigned> category_nums, datapair test_img, unsigned char bin_size) {
    vector<double> prediction(10, 0.0);
    vector<unsigned char> test_bin = count_bins(test_img, bin_size);

    double total = 0;
    double p_bi, p_bni, num_bi;
    double postsum = 0;

    for (long unsigned i = 0; i < category_nums.size(); i++) {
        total += category_nums[i];
    }
    
    for (long unsigned i = 0; i < train_discrete_model.size(); i++) {
        p_bi = p_bni = 0;
        for (long unsigned bin = 0; bin < test_bin.size(); bin++) {
            num_bi = train_discrete_model[i][bin][test_bin[bin]];
            p_bi += log(num_bi / (category_nums[i]));
        }
        p_bi += log(category_nums[i] / total);
        prediction[i] = p_bi;
        postsum += p_bi;
    }

    for (long unsigned i = 0; i < prediction.size(); i++)
        prediction[i] /= postsum;

    return prediction;
}

vector<double> count_continuous_posterior(vector<vector<vector<double> > >& train_continuous_model, vector<unsigned> category_nums, datapair test_img) {
    vector<double> prediction(10, 0.0);

    double total = 0;
    double p_bi, num_bi;
    double postsum = 0;

    for (long unsigned i = 0; i < category_nums.size(); i++) {
        total += category_nums[i];
    }
    
    for (long unsigned i = 0; i < train_continuous_model.size(); i++) {
        p_bi = 0;
        for (long unsigned bin = 0; bin < train_continuous_model[i].size(); bin++) {
            num_bi = train_continuous_model[i][bin][test_img.img[bin]];
            p_bi += log(num_bi);
        }
        p_bi += log(category_nums[i] / total);
        prediction[i] = p_bi;
        postsum += p_bi;
    }

    for (long unsigned i = 0; i < prediction.size(); i++)
        prediction[i] /= postsum;

    return prediction;
}
int judge_pred(vector<double> pred, unsigned char ans) {
    cout << "Posterior (probabilities)" << endl;

    double min = 10e10;
    unsigned char pre = -1;
    for (long unsigned i = 0; i < pred.size(); i++) {
        cout << (unsigned)i << ": " << pred[i] << endl;
        if (pred[i] < min) {
            min = pred[i];
            pre = i;
        }
    }
    cout << "Prediction: " << (unsigned)pre << ", Ans: " << (unsigned)ans << endl << endl;
    return pre == ans ? 0 : 1;
}

int main(int argc, char * argv[]) {
    const string fname_train_img = "train-images.idx3-ubyte";
    const string fname_train_label = "train-labels.idx1-ubyte";
    const string fname_test_img = "t10k-images.idx3-ubyte";
    const string fname_test_label = "t10k-labels.idx1-ubyte";

    int mode;
    cout << "input option (0 for discrete mode, 1 for continuous mode) : ";
    cin >> mode;

    vector<datapair> train_data =  read_datapairs(fname_train_img, fname_train_label);
    vector<datapair> test_data =  read_datapairs(fname_test_img, fname_test_label);

    vector<unsigned> category_nums = count_category_num(train_data);
    vector<vector<vector<double> > > train_model;
    if (mode == 0) {
        train_model = create_discrete_model(train_data, 8, MINIMUM);
    } 

    if (mode == 1) {
        train_model = create_continuous_model(train_data, MINIMUM);
    }

    vector<double> pred;
    double error_rate = 0.0;
    for (long unsigned i = 0; i < test_data.size() ; i++) {
        if (mode == 0) {
            // discrete
            pred = count_discrete_posterior(train_model, category_nums, test_data[i], BIN_SIZE);
        } 
        if (mode == 1) {
            // continuous
            pred = count_continuous_posterior(train_model, category_nums, test_data[i]);
        }
        error_rate += judge_pred(pred, test_data[i].label);
    }
    error_rate /= test_data.size();

    cout << "Imagination of numbers in Bayesian classifier: " << endl;
    for (unsigned long i = 0; i < test_data.size() ; i++) 
        display(test_data[i]);

    cout << "Error rate: " << error_rate << endl;

    return 0;

}


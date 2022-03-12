#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;

#define MINIMUM 1e-10

typedef struct _datapair {
    unsigned char row;
    unsigned char col;
    unsigned char label;
    unsigned char *img = NULL;
} datapair;

unsigned byte2unsigned(char* carr) {
    unsigned ret = 0;
    for (int i = 0; i < 4; i++) {
        ret <<= 8;
        ret |= (unsigned char)carr[i];
    }
    return ret;
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
    for (int i = 0; i < dp_vec.size(); i++)
        ret[dp_vec[i].label] ++;
    return ret;
}

vector<unsigned char> count_bins(datapair &dp) {
    unsigned total = dp.row * dp.col;
    vector<unsigned char> bins(total, 0);
    for (unsigned i = 0; i < total; i++) {
        bins[i] = dp.img[i] / 8;
    }
    return bins;
}

vector<vector<vector<double>>> create_table(vector<datapair>& dp_vec, double init_num=1e-10) {
    vector<vector<vector<double>>> ret(10, vector<vector<double>>(784, vector<double>(32, init_num)));

    for (unsigned i = 0; i < dp_vec.size(); i++) {
        for (unsigned b = 0; b < 784; b++) {
            ret[dp_vec[i].label][b][dp_vec[i].img[b] / 8] += 1;
        }
    }

    return ret;
}

void dump_table(vector<vector<vector<double>>>& train_table){
    for (int i = 0; i < train_table.size(); i++) {
        for (int j = 0; j < train_table[i].size(); j++) {
            for (int k = 0; k < train_table[i][j].size(); k++) {
                cout << train_table[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

vector<double> count_discrete_posterior(vector<vector<vector<double>>>& train_table, vector<unsigned> category_nums, datapair test_img) {
    vector<double> prediction(10, 0.0);
    vector<unsigned char> test_bin = count_bins(test_img);

    double total = 0;
    double p_bi, p_bni, num_bi, num_bni;

    for (int i = 0; i < category_nums.size(); i++) {
        total += category_nums[i];
    }
    
    for (unsigned i = 0; i < train_table.size(); i++) {
        p_bi = p_bni = 1.0;
        for (unsigned bin = 0; bin < test_bin.size(); bin++) {
            num_bi = num_bni = 0.0;
            for (unsigned category = 0; category < train_table.size(); category++) {
                if (i == category) {
                    num_bi = train_table[category][bin][test_bin[bin]];
                } else {
                    num_bni += train_table[category][bin][test_bin[bin]];
                }
            }
            p_bi *= (num_bi / category_nums[i]);
            p_bni *= (num_bni / (total - category_nums[i]));
        }

        p_bi *= (category_nums[i] / total);
        p_bni *= ((total - category_nums[i]) / total);
        prediction[i] = p_bi / (p_bi + p_bni);
        if (prediction[i] < 0.0) {
            cout << "negative" << endl;
            exit(0);
        }
    }

    return prediction;
}

int judge_pred(vector<double> pred, unsigned char ans) {
    cout << "Posterior (probabilities)" << endl;

    double max = -1;
    unsigned char pre = -1;
    for (unsigned char i = 0; i < pred.size(); i++) {
        cout << (unsigned)i << ": " << pred[i] << endl;
        if (pred[i] > max) {
            max = pred[i];
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
    double init_val;
    cout << "input option (0 for discrete mode, 1 for continuous mode) : ";
    cin >> mode;
    if (mode == 0) {
        cout << "input init value : ";
        cin >> init_val;
        init_val = init_val < 0 ? 1e-10 : init_val;
    }

    vector<datapair> train_data =  read_datapairs(fname_train_img, fname_train_label);
    vector<datapair> test_data =  read_datapairs(fname_test_img, fname_test_label);

    vector<unsigned> category_nums = count_category_num(train_data);
    vector<vector<vector<double>>> train_table = create_table(train_data, init_val);

    vector<double> pred;
    double error_rate = 0.0;
    for (int i = 0; i < test_data.size(); i++) {
        pred = count_discrete_posterior(train_table, category_nums, test_data[i]);
        error_rate += judge_pred(pred, test_data[i].label);
    }
    error_rate /= test_data.size();

    cout << "Imagination of numbers in Bayesian classifier: " << endl;
    for (int i = 0; i < test_data.size(); i++) 
        display(test_data[i]);

    cout << "Error rate: " << error_rate << endl;

    return 0;

}


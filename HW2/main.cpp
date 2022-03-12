#include <vector>
#include <iomanip>
#include <fstream>
#include <string>
#include <iostream>
using namespace std;

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
}

void dump_histogram(vector<vector<unsigned>> histograms) {
    for (unsigned i = 0; i < histograms.size(); i++) {
        for (unsigned b = 0; b < 32; b++) {
            cout << setw(5) << histograms[i][b];
        }
        cout << endl;
    }
}

vector<unsigned> count_histogram(datapair &dp) {
    vector<unsigned> histogram(32, 1);
    for (unsigned char i = 0; i < dp.row; i++) {
        for (unsigned char j = 0; j < dp.row; j++) {
            histogram[((unsigned)dp.img[i * dp.row + j] / 8)]++;
        }
    }
    return histogram;
}

vector<vector<vector<unsigned>>> create_prior(vector<datapair> &dp_vec) {
    vector<vector<vector<unsigned>>> ret(10, vector<vector<unsigned>>());

    for (unsigned i = 0; i < dp_vec.size(); i++) {
        ret[dp_vec[i].label].push_back(count_histogram(dp_vec[i]));
    }

    return ret;
} 

int main(int argc, char * argv[]) {
    const string fname_train_img = "train-images.idx3-ubyte";
    const string fname_train_label = "train-labels.idx1-ubyte";
    const string fname_test_img = "t10k-images.idx3-ubyte";
    const string fname_test_label = "t10k-labels.idx1-ubyte";

    vector<datapair> train_data =  read_datapairs(fname_train_img, fname_train_label);
    vector<datapair> test_data =  read_datapairs(fname_test_img, fname_test_label);

    vector<vector<vector<unsigned>>> train_hist_0to9 = create_prior(train_data);
    vector<vector<vector<unsigned>>> test_hist_0to9 = create_prior(test_data);

    dump_histogram(train_hist_0to9[0]);

    display(train_data[0]);
    display(test_data[0]);


    return 0;

}


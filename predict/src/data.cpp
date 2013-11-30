
#include "data.h"

using namespace std;

DataProvider::DataProvider(int minibatchSize) :
    _hData(NULL), _minibatchSize(minibatchSize), _dataSize(0) {

}

Matrix& DataProvider::operator[](int idx) {
    return getMinibatch(idx);
}

void DataProvider::clearData() {
    _hData = NULL;
    _dataSize = 0;
}

void DataProvider::setData(Matrix& hData) {
    _hData = &hData;
    _dataSize = _hData->getNumDataBytes();
}

Matrix& DataProvider::getMinibatch(int idx) {
    assert(idx >= 0 && idx < getNumMinibatches());
    return getDataSlice(idx * _minibatchSize, (idx + 1) * _minibatchSize);
}

Matrix& DataProvider::getDataSlice(int startCase, int endCase) {
    assert(_hData != NULL);

    Matrix *miniData = new Matrix();

    if (_hData->isTrans()) {
        _hData->sliceCols(startCase, min(getNumCases(), endCase), *miniData);
    } else {
        _hData->sliceRows(startCase, min(getNumCases(), endCase), *miniData);
    }

    return *miniData;
}

int DataProvider::getNumCases() {
    assert(_hData != NULL);

    return _hData->getFollowingDim();
}

int DataProvider::getNumMinibatches() {
    assert(_hData != NULL);
    assert(getNumCases() > 0);
    assert(_minibatchSize > 0);

    return getNumCases() / _minibatchSize;
}

int DataProvider::getMinibatchSize() {
    return _minibatchSize;
}

int DataProvider::getNumCasesInMinibatch(int idx) {
    assert(_hData != NULL);
    assert(getNumCases() > 0);
    assert(idx >= 0 && idx < getNumMinibatches());

    return min(_minibatchSize, max(0, getNumCases() - idx * _minibatchSize));
}

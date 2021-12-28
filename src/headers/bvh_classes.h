//class BVHNode {
//public:
//    BVHNode *top1, *top2, *top3, *top4, *bottom1, *bottom2, *bottom3, *bottom4;
//    int objectIndex;
//    BVHNode() {
//
//    }
//};

struct Bounds {
public:
    float top, bottom, left, right, front, back;
    Bounds(): top(-9999), bottom(9999), left(9999), right(-9999), front(-9999), back(9999) {}
    Bounds(float _t, float _b, float _l, float _r, float _f, float _back): top(_t), bottom(_b),
    left(_l), right(_r), front(_f), back(_back) {}

    void reset() {
        top = -9999; bottom = 9999; left = 9999; right = -9999; front = -9999; back = 9999;
    }
};

class BVHBinaryNode {
public:
    BVHBinaryNode *left, *right;
    Bounds* bounds;
    int* objectsIndex;
    int numObjects;
    BVHBinaryNode(): numObjects(0), bounds(new Bounds()), left(nullptr), right(nullptr), objectsIndex(nullptr) {}
    BVHBinaryNode(Bounds* _bounds): bounds(_bounds), left(nullptr), right(nullptr), objectsIndex(nullptr), numObjects(0) {}
    BVHBinaryNode(Bounds* _bounds, int* _objectsIndex, int _numObject): bounds(_bounds), objectsIndex(_objectsIndex),
                                                                        left(nullptr), right(nullptr), numObjects(_numObject) {}
    BVHBinaryNode(Bounds* _bounds, BVHBinaryNode* _left, BVHBinaryNode* _right): bounds(_bounds),
                                                                                 left(_left), right(_right), objectsIndex(nullptr), numObjects(0) {}
};

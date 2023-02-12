#ifndef SCREENH
#define SCREENH

/*---------------------------------CLASS DEFINITION--------------------------------*/
class Screen {
    public:
    /*----------------------------------ATTRIBUTES---------------------------------*/
    float width, height;

    /*--------------------------------CONSTRUCTORS---------------------------------*/
    __host__ Screen();
    __host__ Screen(float w, float h);
    __host__ Screen(const Screen &screen);
    __host__ ~Screen() = default;
};

/*--------------------------------CONSTRUCTORS---------------------------------*/
__host__ Screen::Screen() { width = 640.0, height = 480.0; }
__host__ Screen::Screen(float w, float h) { width = w, height = h; }
__host__ Screen::Screen(const Screen &screen) { *this = screen; }

#endif
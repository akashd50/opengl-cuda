class MainOpenGL {
public: 
	static const char* WINDOW_TITLE;
	static const double FRAME_RATE_MS;
	static int WIDTH;
	static int HEIGHT;

	static void init(void);
	static void update(void);
	static void display(void);
	static void keyboard(unsigned char key, int x, int y);
	static void mouse(int button, int state, int x, int y);
	static void reshape(int width, int height);
};
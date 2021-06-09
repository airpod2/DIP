
#include "panorama.h"

void ex_panorama_simple() {
	Mat img;
	vector<Mat> imgs, imgs2;
	img = imread("left.jpg", IMREAD_COLOR);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	imgs.push_back(img);

	img = imread("center.jpg", IMREAD_COLOR);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	imgs.push_back(img);

	Mat result, final_result;
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, false); //try_use_gpu=false
	Stitcher::Status status = stitcher->stitch(imgs, result);
	if (status != Stitcher::OK) {
		cout << "Can't stitch images, error code" << int(status) << endl; //공통되는 영역이 너무 적으면 eror code 1을 출력
		exit(-1);
	}
	imgs2.push_back(result);
	img = imread("right.jpg", IMREAD_COLOR);
	if (img.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	imgs2.push_back(img);
	status = stitcher->stitch(imgs2, final_result);
	imshow("ex_panorama_simple 3", final_result);
	imwrite("ex_panorama_simple.jpg", final_result);
	waitKey();
}

Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
	// < gray scale로 변환 >
	Mat img_gray_l, img_gray_r;
	cvtColor(img_l, img_gray_l, CV_BGR2GRAY);
	cvtColor(img_r, img_gray_r, CV_BGR2GRAY);

	// < 특징점(key point) 추출 >
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(img_gray_l, kpts_obj);
	Detector->detect(img_gray_r, kpts_scene);

	// < 특징점 시각화 >
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imwrite("img_kpts_l.png", img_kpts_l);
	imwrite("img_kpts_r.png", img_kpts_r);

	// < 기술자(descriptor) 추출 >
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);
	Mat img_des_obj, img_des_scene;
	Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
	Extractor->compute(img_gray_r, kpts_scene, img_des_scene);

	// < 기술자를 이용한 특징점 매칭 >
	BFMatcher matcher(NORM_L2); //Brute Force 매칭
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	// < 매칭결과 시각화 >
	Mat img_matches;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches.png", img_matches);

	//매칭결과 정제
	//매칭거리가 작은 우수한 매칭 결과를 정제하는 과정
	//최소 매칭 거리의 3배 또는 우수한 매칭 결과 60이상까지 정제

	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min)dist_min = dist;
		if (dist > dist_max)dist_max = dist;
	}
	printf("max_dist :  %f\n", dist_max); //max는 사실상 불필요
	printf("min_dist :  %f\n", dist_min);
	
	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	// < 우수한 매칭 결과 시각화 >
	Mat img_matches_good;
	drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imwrite("img_matches_good.png", img_matches_good);

	// < 매칭 결과 좌표 추출 >
	vector<Point2f>obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); //img2
	}

	// < 매칭 결과로부터 homograpy 행렬을 추출 >
	Mat mat_homo = findHomography(scene, obj, RANSAC); //이상치(outlier)제거를 위해 RANSAC추가

	// < Homograpy 행렬을 이용해 시점 역변환 >
	Mat img_result;
	warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC); //영상이 잘리는 것을 방지하기 위해 여유공간을 부여

	// < 기준 영상과 역변환된 시점 영상 합체 >
	Mat img_pano;
	img_pano = img_result.clone();
	Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
	img_l.copyTo(roi);

	// < 검은 여백 잘라내기 >
	int cut_x = 0, cut_y = 0;
	for (int y = 0; y < img_pano.rows; y++) {
		for (int x = 0; x < img_pano.cols; x++) {
			if (img_pano.at<Vec3b>(y, x)[0] == 0 && img_pano.at<Vec3b>(y, x)[1] == 0 && img_pano.at<Vec3b>(y, x)[2] == 0)
				continue;

			if (cut_x < x) cut_x = x;
			if (cut_y < y) cut_y = y;
		}
	}
	Mat img_pano_cut;
	img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
	imwrite("img_pano_cut.png", img_pano_cut);

	return img_pano_cut;
}
void ex_panorama() {
	Mat matImage1 = imread("center.jpg", IMREAD_COLOR);
	resize(matImage1, matImage1, Size(), 0.3, 0.3);
	Mat matImage2 = imread("left.jpg", IMREAD_COLOR);
	resize(matImage2, matImage2, Size(), 0.3, 0.3);
	Mat matImage3 = imread("right.jpg", IMREAD_COLOR);
	resize(matImage3, matImage3, Size(), 0.3, 0.3);
	if (matImage1.empty() || matImage2.empty() || matImage3.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	Mat result;
	flip(matImage1, matImage1, 1);
	flip(matImage2, matImage2, 1);
	result = makePanorama(matImage1, matImage2, 3, 60); //60, 2:50
	flip(result, result, 1);
	result = makePanorama(result, matImage3, 3, 60);

	imwrite("my_ex_panorama_result2.jpg", result);
	resize(result, result, Size(), 0.8, 0.8);
	imshow("ex_panorama_result", result);
	
	waitKey();
}

void BooknScene() {
	// < gray scale로 변환 >
	Mat book = imread("Book1.jpg", IMREAD_COLOR);
	Mat book_scene = imread("Scene.jpg", IMREAD_COLOR);
	if (book.empty() || book_scene.empty()) {
		cout << "Empty image!\n";
		exit(-1);
	}
	resize(book, book, Size(), 0.5, 0.5);
	resize(book_scene, book_scene, Size(), 0.5, 0.5);
	Mat Book_gray, Scene_gray;
	cvtColor(book, Book_gray, CV_BGR2GRAY);
	cvtColor(book_scene, Scene_gray, CV_BGR2GRAY);

	// < 특징점(key point) 추출  sift
	//Ptr<cv::SiftFeatureDetector>  Detector = SiftFeatureDetector::create();
	Ptr<SurfFeatureDetector> Detector = SURF::create(300);
	vector<KeyPoint> kpts_obj, kpts_scene;
	Detector->detect(Book_gray, kpts_obj);
	Detector->detect(Scene_gray, kpts_scene);

	// < 특징점 시각화 >
	Mat img_kpts_l, img_kpts_r;
	drawKeypoints(Book_gray, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(Scene_gray, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	// < 기술자(descriptor) 추출 >
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);
	Mat img_des_obj, img_des_scene;
	Extractor->compute(Book_gray, kpts_obj, img_des_obj);
	Extractor->compute(Scene_gray, kpts_scene, img_des_scene);

	// < 기술자를 이용한 특징점 매칭 >
	BFMatcher matcher(NORM_L2); //Brute Force 매칭
	vector<DMatch> matches;
	matcher.match(img_des_obj, img_des_scene, matches);

	// < 매칭결과 시각화 >
	Mat img_matches;
	drawMatches(Book_gray, kpts_obj, Scene_gray, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	//< 매칭결과 정제 >
	//매칭거리가 작은 우수한 매칭 결과를 정제하는 과정
	//최소 매칭 거리의 3배 또는 우수한 매칭 결과 60이상까지 정제

	double dist_max = matches[0].distance;
	double dist_min = matches[0].distance;
	double dist;
	for (int i = 0; i < img_des_obj.rows; i++) {
		dist = matches[i].distance;
		if (dist < dist_min)dist_min = dist;
		if (dist > dist_max)dist_max = dist;
	}
	printf("max_dist :  %f\n", dist_max); //max는 사실상 불필요
	printf("min_dist :  %f\n", dist_min);

	int thresh_dist = 3;
	int min_matches = 60;
	vector<DMatch> matches_good;
	do {
		vector<DMatch> good_matches2;
		for (int i = 0; i < img_des_obj.rows; i++) {
			if (matches[i].distance < thresh_dist * dist_min)
				good_matches2.push_back(matches[i]);
		}
		matches_good = good_matches2;
		thresh_dist -= 1;
	} while (thresh_dist != 2 && matches_good.size() > min_matches);

	// < 우수한 매칭 결과 시각화 >
	Mat img_matches_good;
	drawMatches(Book_gray, kpts_obj, Scene_gray, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//imwrite("img_matches_good.png", img_matches_good);

	// < 매칭 결과 좌표 추출 >
	vector<Point2f>obj, scene;
	for (int i = 0; i < matches_good.size(); i++) {
		obj.push_back(kpts_obj[matches_good[i].queryIdx].pt); //img1
		scene.push_back(kpts_scene[matches_good[i].trainIdx].pt); //img2
	}

	// < 매칭 결과로부터 homograpy 행렬을 추출 >
	Mat mat_homo = findHomography(scene, obj, RANSAC); //이상치(outlier)제거를 위해 RANSAC추가

	// < Homograpy 행렬을 이용해 시점 역변환 >
	Mat img_result;
	warpPerspective(book_scene, img_result, mat_homo, Size(book.cols * 2, book.rows * 1.2), INTER_CUBIC); //영상이 잘리는 것을 방지하기 위해 여유공간을 부여


	imshow("img_matches_good book and Scene", img_matches_good);
	imshow("img_result book and Scene", img_result);

	/*imshow("book", img_kpts_l);
	imshow("scene", img_kpts_r);
	imshow("img_matches book and scene", img_matches);*/
	waitKey();
	destroyAllWindows();
}
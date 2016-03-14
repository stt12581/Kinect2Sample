// HDFace.cpp : コンソール アプリケーションのエントリ ポイントを定義します。
// This source code is licensed under the MIT license. Please see the License in License.txt.
// "This is preliminary software and/or hardware and APIs are preliminary and subject to change."
//

#include "stdafx.h"
#include <Windows.h>
#include <Kinect.h>
#include <Kinect.Face.h>
#include <opencv2/opencv.hpp>
#include <bitset>
#include <vector>
#include <fstream>
#include "C:\Users\z4shang\Documents\libsvm-master\svm.h"

using namespace std;
const vector<double> DrawFaceFrameResults(const CameraSpacePoint* pHeadPivot, const float* pAnimUnits, double initialVal[], bool begin, int count, vector<vector<double>> &data);
void signInitialFaceValue(double initialVal[]);
void readTrainingData(string fileName, svm_problem* problem);
void setNewParameter(svm_parameter& parameter);
void getEigenVec(double eigenVec[][3]);

#define TOTAL_INIT 210
#define TOTAL_DATA 840

template<class Interface>
inline void SafeRelease( Interface *& pInterfaceToRelease )
{
	if( pInterfaceToRelease != NULL ){
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}

int _tmain( int argc, _TCHAR* argv[] )
{
	cv::setUseOptimized( true );

	// Sensor
	IKinectSensor* pSensor;
	HRESULT hResult = S_OK;
	hResult = GetDefaultKinectSensor( &pSensor );
	if( FAILED( hResult ) ){
		std::cerr << "Error : GetDefaultKinectSensor" << std::endl;
		return -1;
	}

	hResult = pSensor->Open();
	if( FAILED( hResult ) ){
		std::cerr << "Error : IKinectSensor::Open()" << std::endl;
		return -1;
	}

	// Source
	IColorFrameSource* pColorSource;
	hResult = pSensor->get_ColorFrameSource( &pColorSource );
	if( FAILED( hResult ) ){
		std::cerr << "Error : IKinectSensor::get_ColorFrameSource()" << std::endl;
		return -1;
	}

	IBodyFrameSource* pBodySource;
	hResult = pSensor->get_BodyFrameSource( &pBodySource );
	if( FAILED( hResult ) ){
		std::cerr << "Error : IKinectSensor::get_BodyFrameSource()" << std::endl;
		return -1;
	}

	// Reader
	IColorFrameReader* pColorReader;
	hResult = pColorSource->OpenReader( &pColorReader );
	if( FAILED( hResult ) ){
		std::cerr << "Error : IColorFrameSource::OpenReader()" << std::endl;
		return -1;
	}

	IBodyFrameReader* pBodyReader;
	hResult = pBodySource->OpenReader( &pBodyReader );
	if( FAILED( hResult ) ){
		std::cerr << "Error : IBodyFrameSource::OpenReader()" << std::endl;
		return -1;
	}

	// Description
	IFrameDescription* pDescription;
	hResult = pColorSource->get_FrameDescription( &pDescription );
	if( FAILED( hResult ) ){
		std::cerr << "Error : IColorFrameSource::get_FrameDescription()" << std::endl;
		return -1;
	}

	int width = 0;
	int height = 0;
	pDescription->get_Width( &width ); // 1920
	pDescription->get_Height( &height ); // 1080
	unsigned int bufferSize = width * height * 4 * sizeof( unsigned char );

	cv::Mat bufferMat( height, width, CV_8UC4 );
	cv::Mat faceMat( height / 2, width / 2, CV_8UC4 );
	cv::namedWindow( "HDFace" );

	// Color Table
	cv::Vec3b color[BODY_COUNT];
	color[0] = cv::Vec3b( 255, 0, 0 );
	color[1] = cv::Vec3b( 0, 255, 0 );
	color[2] = cv::Vec3b( 0, 0, 255 );
	color[3] = cv::Vec3b( 255, 255, 0 );
	color[4] = cv::Vec3b( 255, 0, 255 );
	color[5] = cv::Vec3b( 0, 255, 255 );

	// Coordinate Mapper
	ICoordinateMapper* pCoordinateMapper;
	hResult = pSensor->get_CoordinateMapper( &pCoordinateMapper );
	if( FAILED( hResult ) ){
		std::cerr << "Error : IKinectSensor::get_CoordinateMapper()" << std::endl;
		return -1;
	}

	IHighDefinitionFaceFrameSource* pHDFaceSource[BODY_COUNT];
	IHighDefinitionFaceFrameReader* pHDFaceReader[BODY_COUNT];
	double initialFaceValue[FaceShapeAnimations_Count];
	vector<vector<double>> data;
	//signInitialFaceValue(initialFaceValue);
	bool begin = true;
	int initial_count = 0;
	svm_problem problem;
	
	//read training dataset
	ifstream inFile;
	problem.y = new double[FaceShapeAnimations_Count];
	for (int m = 0; m < FaceShapeAnimations_Count; m++)
	problem.x = new svm_node*[TOTAL_DATA];
	problem.l = TOTAL_DATA;
	for (int i = 0; i < TOTAL_DATA; i++) problem.x[i] = new svm_node[4];

	inFile.open("newData.txt");
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < TOTAL_DATA; j++){
			problem.x[j][i].index = i;
			inFile >> problem.x[j][i].value;
		}
	}
	for (int i = 0; i < TOTAL_DATA; i++){
		problem.x[i][3].index = -1;
		if (i<210) problem.y[i] = 0;
		else if (i<420) problem.y[i] = 1;
		else if (i<630) problem.y[i] = 2;
		else problem.y[i] = 3;
	}
	inFile.close();

	double eigenVec[17][3];
	getEigenVec(eigenVec);

	//set new parameter
	struct svm_parameter parameter;
	parameter.svm_type = C_SVC;
	parameter.kernel_type = RBF;
	parameter.gamma = 0.3333;
	parameter.degree = 3;
	parameter.cache_size = 40;
	parameter.eps = 0.1;
	parameter.C = 1;
	parameter.shrinking = 1;
	parameter.probability = 0;
	parameter.nr_weight = 0;
	const char* result = svm_check_parameter(&problem, &parameter);
	if (result != NULL){
		cout << "haha"<<result << endl;
		return 0;
	}
	svm_model* model = svm_train(&problem, &parameter);

	IFaceModelBuilder* pFaceModelBuilder[BODY_COUNT];
	bool produce[BODY_COUNT] = { false };
	IFaceAlignment* pFaceAlignment[BODY_COUNT];
	IFaceModel* pFaceModel[BODY_COUNT];
	std::vector<std::vector<float>> deformations( BODY_COUNT, std::vector<float>( FaceShapeDeformations::FaceShapeDeformations_Count ) );
	for( int count = 0; count < BODY_COUNT; count++ ){
		// Source
		hResult = CreateHighDefinitionFaceFrameSource( pSensor, &pHDFaceSource[count] );
		if( FAILED( hResult ) ){
			std::cerr << "Error : CreateHighDefinitionFaceFrameSource()" << std::endl;
			return -1;
		}

		// Reader
		hResult = pHDFaceSource[count]->OpenReader( &pHDFaceReader[count] );
		if( FAILED( hResult ) ){
			std::cerr << "Error : IHighDefinitionFaceFrameSource::OpenReader()" << std::endl;
			return -1;
		}

		// Open Face Model Builder
		hResult = pHDFaceSource[count]->OpenModelBuilder( FaceModelBuilderAttributes::FaceModelBuilderAttributes_None, &pFaceModelBuilder[count] );
		if( FAILED( hResult ) ){
			std::cerr << "Error : IHighDefinitionFaceFrameSource::OpenModelBuilder()" << std::endl;
			return -1;
		}

		// Start Collection Face Data
		hResult = pFaceModelBuilder[count]->BeginFaceDataCollection();
		if( FAILED( hResult ) ){
			std::cerr << "Error : IFaceModelBuilder::BeginFaceDataCollection()" << std::endl;
			return -1;
		}

		// Create Face Alignment
		hResult = CreateFaceAlignment( &pFaceAlignment[count] );
		if( FAILED( hResult ) ){
			std::cerr << "Error : CreateFaceAlignment()" << std::endl;
			return -1;
		}

		// Create Face Model
		hResult = CreateFaceModel( 1.0f, FaceShapeDeformations::FaceShapeDeformations_Count, &deformations[count][0], &pFaceModel[count] );
		if( FAILED( hResult ) ){
			std::cerr << "Error : CreateFaceModel()" << std::endl;
			return -1;
		}
	}

	UINT32 vertex = 0;
	hResult = GetFaceModelVertexCount( &vertex ); // 1347
	if( FAILED( hResult ) ){
		std::cerr << "Error : GetFaceModelVertexCount()" << std::endl;
		return -1;
	}

	while( 1 ){
		// Color Frame
		IColorFrame* pColorFrame = nullptr;
		hResult = pColorReader->AcquireLatestFrame( &pColorFrame );
		if( SUCCEEDED( hResult ) ){
			hResult = pColorFrame->CopyConvertedFrameDataToArray( bufferSize, reinterpret_cast<BYTE*>( bufferMat.data ), ColorImageFormat::ColorImageFormat_Bgra );
			if( SUCCEEDED( hResult ) ){
				cv::resize( bufferMat, faceMat, cv::Size(), 0.5, 0.5 );
			}
		}
		SafeRelease( pColorFrame );

		// Body Frame
		IBodyFrame* pBodyFrame = nullptr;
		hResult = pBodyReader->AcquireLatestFrame( &pBodyFrame );
		if( SUCCEEDED( hResult ) ){
			IBody* pBody[BODY_COUNT] = { 0 };
			hResult = pBodyFrame->GetAndRefreshBodyData( BODY_COUNT, pBody );
			if( SUCCEEDED( hResult ) ){
				for( int count = 0; count < BODY_COUNT; count++ ){
					BOOLEAN bTrackingIdValid = false;
					hResult = pHDFaceSource[count]->get_IsTrackingIdValid( &bTrackingIdValid );
					if( !bTrackingIdValid ){
						BOOLEAN bTracked = false;
						hResult = pBody[count]->get_IsTracked( &bTracked );
						if( SUCCEEDED( hResult ) && bTracked ){
							/*// Joint
							Joint joint[JointType::JointType_Count];
							hResult = pBody[count]->GetJoints( JointType::JointType_Count, joint );
							if( SUCCEEDED( hResult ) ){
								for( int type = 0; type < JointType::JointType_Count; type++ ){
									ColorSpacePoint colorSpacePoint = { 0 };
									pCoordinateMapper->MapCameraPointToColorSpace( joint[type].Position, &colorSpacePoint );
									int x = static_cast<int>( colorSpacePoint.X );
									int y = static_cast<int>( colorSpacePoint.Y );
									if( ( x >= 0 ) && ( x < width ) && ( y >= 0 ) && ( y < height ) ){
										cv::circle( bufferMat, cv::Point( x, y ), 5, static_cast<cv::Scalar>( color[count] ), -1, CV_AA );
									}
								}
							}*/

							// Set TrackingID to Detect Face
							UINT64 trackingId = _UI64_MAX;
							hResult = pBody[count]->get_TrackingId( &trackingId );
							if( SUCCEEDED( hResult ) ){
								pHDFaceSource[count]->put_TrackingId( trackingId );
							}
						}
					}
				}
			}
			for( int count = 0; count < BODY_COUNT; count++ ){
				SafeRelease( pBody[count] );
			}
		}
		SafeRelease( pBodyFrame );

		// HD Face Frame
		for( int count = 0; count < BODY_COUNT; count++ ){
			IHighDefinitionFaceFrame* pHDFaceFrame = nullptr;
			hResult = pHDFaceReader[count]->AcquireLatestFrame( &pHDFaceFrame );
			if( SUCCEEDED( hResult ) && pHDFaceFrame != nullptr ){
				BOOLEAN bFaceTracked = false;
				hResult = pHDFaceFrame->get_IsFaceTracked( &bFaceTracked );
				if( SUCCEEDED( hResult ) && bFaceTracked ){
					hResult = pHDFaceFrame->GetAndRefreshFaceAlignmentResult( pFaceAlignment[count] );
					if( SUCCEEDED( hResult ) && pFaceAlignment[count] != nullptr ){
						// Face Model Building
						if( !produce[count] ){
							std::system( "cls" );
							FaceModelBuilderCollectionStatus collection;
							hResult = pFaceModelBuilder[count]->get_CollectionStatus( &collection );
							if( collection == FaceModelBuilderCollectionStatus::FaceModelBuilderCollectionStatus_Complete ){
								std::cout << "Status : Complete" << std::endl;
								cv::putText( bufferMat, "Status : Complete", cv::Point( 50, 50 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
								IFaceModelData* pFaceModelData = nullptr;
								hResult = pFaceModelBuilder[count]->GetFaceData( &pFaceModelData );
								if( SUCCEEDED( hResult ) && pFaceModelData != nullptr ){
									/*hResult = pFaceModelData->ProduceFaceModel( &pFaceModel[count] );
									if( SUCCEEDED( hResult ) && pFaceModel[count] != nullptr ){
										produce[count] = true;
									}*/

									CameraSpacePoint headPivot;
									pFaceAlignment[count]->get_HeadPivotPoint(&headPivot);
									float* pAnimationUnits = new float[FaceShapeAnimations_Count];
									pFaceAlignment[count]->GetAnimationUnits(FaceShapeAnimations_Count, pAnimationUnits);

									// draw face frame results
									/*
									    begin = true, add value to initial
										begin = false && initial_count != 0, calculate average initial value
										begin = false && initial_count ==0, do regular recognition
									*/
									vector<double> testData_vec= DrawFaceFrameResults(&headPivot, pAnimationUnits, initialFaceValue, begin, initial_count, data);
									svm_node* testData = new svm_node[4];
									
									for (int i = 0; i < 3; i++){
										testData[i].value = 0;
										for (int j = 0; j < FaceShapeAnimations_Count; j++){
											testData[i].value += testData_vec[j] * eigenVec[j][i];
										}
										testData[i].index = i;
									}
									testData[3].index = -1;
									/*
									for (int i = 0; i < FaceShapeAnimations_Count; i++){
										testData[i].value = testData_vec[i];
										testData[i].index = i;
									}
									testData[17].index = -1;*/
									
									if (!begin) initial_count = 0;
									else{
										initial_count++;
										if (initial_count >= TOTAL_INIT){
											begin = false;
											ofstream myfile;
											myfile.open("hahahaha.txt");
											for (int i = 0; i < data.size(); i++){
												for (int j = 0; j < data[0].size(); j++){
													myfile << data[i][j] << " ";
												}
												myfile << endl;
											}
											myfile.close();
											
										}
									}

									
									double res = svm_predict(model, testData);
									wstring faceText;
									string outputResult = "";
									if (res == 0){
										outputResult = ":-|";
										faceText = L":-|\n";
									}
									else if (res == 1){
										outputResult = ":-)";
										faceText = L":-)\n";
									}
									else if (res == 2){
										outputResult = ":-O";
										faceText = L":-O\n";
									}
									else{
										outputResult = ":-(";
										faceText = L":-(\n";
									}
									wcout << faceText << endl;
									cv::putText(bufferMat, outputResult, cv::Point(50, 200), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>(color[count]), 2, CV_AA);
									delete[] pAnimationUnits;
								}

								SafeRelease( pFaceModelData );
							}
							else{
								std::cout << "Status : " << collection << std::endl;
								cv::putText( bufferMat, "Status : " + std::to_string( collection ), cv::Point( 50, 50 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );

								// Collection Status
								if( collection >= FaceModelBuilderCollectionStatus::FaceModelBuilderCollectionStatus_TiltedUpViewsNeeded ){
									std::cout << "Need : Tilted Up Views" << std::endl;
									cv::putText( bufferMat, "Need : Tilted Up Views", cv::Point( 50, 100 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
								}
								else if( collection >= FaceModelBuilderCollectionStatus::FaceModelBuilderCollectionStatus_RightViewsNeeded ){
									std::cout << "Need : Right Views" << std::endl;
									cv::putText( bufferMat, "Need : Right Views", cv::Point( 50, 100 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
								}
								else if( collection >= FaceModelBuilderCollectionStatus::FaceModelBuilderCollectionStatus_LeftViewsNeeded ){
									std::cout << "Need : Left Views" << std::endl;
									cv::putText( bufferMat, "Need : Left Views", cv::Point( 50, 100 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
								}
								else if( collection >= FaceModelBuilderCollectionStatus::FaceModelBuilderCollectionStatus_FrontViewFramesNeeded ){
									std::cout << "Need : Front ViewFrames" << std::endl;
									cv::putText( bufferMat, "Need : Front ViewFrames", cv::Point( 50, 100 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
								}

								// Capture Status
								FaceModelBuilderCaptureStatus capture;
								hResult = pFaceModelBuilder[count]->get_CaptureStatus( &capture );
								switch( capture ){
									case FaceModelBuilderCaptureStatus::FaceModelBuilderCaptureStatus_FaceTooFar:
										std::cout << "Error : Face Too Far from Camera" << std::endl;
										cv::putText( bufferMat, "Error : Face Too Far from Camera", cv::Point( 50, 150 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
										break;
									case FaceModelBuilderCaptureStatus::FaceModelBuilderCaptureStatus_FaceTooNear:
										std::cout << "Error : Face Too Near to Camera" << std::endl;
										cv::putText( bufferMat, "Error : Face Too Near to Camera", cv::Point( 50, 150 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
										break;
									case FaceModelBuilderCaptureStatus_MovingTooFast:
										std::cout << "Error : Moving Too Fast" << std::endl;
										cv::putText( bufferMat, "Error : Moving Too Fast", cv::Point( 50, 150 ), cv::FONT_HERSHEY_SIMPLEX, 1.0f, static_cast<cv::Scalar>( color[count] ), 2, CV_AA );
										break;
									default:
										break;
								}
							}
						}

						// HD Face Points
						std::vector<CameraSpacePoint> facePoints( vertex );
						hResult = pFaceModel[count]->CalculateVerticesForAlignment( pFaceAlignment[count], vertex, &facePoints[0] );
						if( SUCCEEDED( hResult ) ){
							for( int point = 0; point < vertex; point++ ){
								ColorSpacePoint colorSpacePoint;
								hResult = pCoordinateMapper->MapCameraPointToColorSpace( facePoints[point], &colorSpacePoint );
								if( SUCCEEDED( hResult ) ){
									int x = static_cast<int>( colorSpacePoint.X );
									int y = static_cast<int>( colorSpacePoint.Y );
									if( ( x >= 0 ) && ( x < width ) && ( y >= 0 ) && ( y < height ) ){
										cv::circle( bufferMat, cv::Point( static_cast<int>( colorSpacePoint.X ), static_cast<int>( colorSpacePoint.Y ) ), 2, static_cast<cv::Scalar>( color[count] ), -1, CV_AA );
									}
								}
							}
						}
					}
				}
			}
			SafeRelease( pHDFaceFrame );
		}

		cv::resize( bufferMat, faceMat, cv::Size(), 0.5, 0.5 );
		cv::imshow( "HDFace", faceMat );

		if( cv::waitKey( 10 ) == VK_ESCAPE ){
			break;
		}
	}

	SafeRelease( pColorSource );
	SafeRelease( pBodySource );
	SafeRelease( pColorReader );
	SafeRelease( pBodyReader );
	SafeRelease( pDescription );
	SafeRelease( pCoordinateMapper );
	for( int count = 0; count < BODY_COUNT; count++ ){
		SafeRelease( pHDFaceSource[count] );
		SafeRelease( pHDFaceReader[count] );
		SafeRelease( pFaceModelBuilder[count] );
		SafeRelease( pFaceAlignment[count] );
		SafeRelease( pFaceModel[count] );
	}
	if( pSensor ){
		pSensor->Close();
	}
	SafeRelease( pSensor );
	cv::destroyAllWindows();

	return 0;
}

const vector<double> DrawFaceFrameResults(const CameraSpacePoint* pHeadPivot, const float* pAnimUnits, double initialVal[], bool begin, int count, vector<vector<double>> &data)
{

		std::wstring faceText = L"";

		faceText += L"HeadPivot Coordinates\n";
		faceText += L" X-> " + std::to_wstring(pHeadPivot->X) + L" Y-> " + std::to_wstring(pHeadPivot->Y) + L" Z-> " + std::to_wstring(pHeadPivot->Z) + L" \n";
		double percent[FaceShapeAnimations_Count];
		vector<double> subData;
		vector<double> testData;

		//get the HDFace animation units
		for (int i = 0; i < FaceShapeAnimations_Count; i++)
		{
			FaceShapeAnimations faceAnim = (FaceShapeAnimations)i;
			std::wstring strValue = std::to_wstring(pAnimUnits[faceAnim]);
			if (pAnimUnits[faceAnim] == 0) percent[i] = 0;
			else percent[i] = (pAnimUnits[faceAnim] - initialVal[i]) / pAnimUnits[faceAnim];

				switch (faceAnim)
				{
				case FaceShapeAnimations::FaceShapeAnimations_JawOpen:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim]/count;
					faceText += L"JawOpened: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_JawSlideRight:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"JawSlideRight: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LeftcheekPuff:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LeftCheekPuff: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LefteyebrowLowerer:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LeftEyeBrowLowered: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LefteyeClosed:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LeftEyeClosed: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_RighteyebrowLowerer:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"RightEyeBrowLowered: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_RighteyeClosed:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"RightEyeClosed: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipCornerDepressorLeft:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LipCornerDepressedLeft: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipCornerDepressorRight:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LipCornerDepressedRight: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipCornerPullerLeft:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LipCornerPulledLeft: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipCornerPullerRight:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LipCornerPulledRight: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipPucker:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"Lips Puckered: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipStretcherLeft:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LipStretchLeft: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LipStretcherRight:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LipStretchRight: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LowerlipDepressorLeft:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LowerLipDepressedLeft: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_LowerlipDepressorRight:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"LowerLipDepressedRight: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;
				case FaceShapeAnimations::FaceShapeAnimations_RightcheekPuff:
					if (begin){
						initialVal[faceAnim] += pAnimUnits[faceAnim];
						subData.push_back(pAnimUnits[faceAnim]);
					}
					testData.push_back(pAnimUnits[faceAnim]);
					if (!begin && count != 0) initialVal[faceAnim] = initialVal[faceAnim] / count;
					faceText += L"RightCheekPuffed: " + strValue + L"    init: " + to_wstring(initialVal[faceAnim]) + L"  %: " + to_wstring(percent[i]) + L"\n";
					break;

				}
				
		}
		data.push_back(subData);
		
		if (begin){
			wcout << "hahaha" << endl;
		}
		else{
			wcout << faceText << endl;
		}
		/*
		if (begin){
			wcout << "hahaha" << endl;
			return 0;
		}

		if (percent[FaceShapeAnimations::FaceShapeAnimations_LipStretcherLeft] > 0.4 ||
			percent[FaceShapeAnimations::FaceShapeAnimations_LeftcheekPuff] > 0.6){
			faceText += L":-)\n";
			wcout << faceText << endl;
			return 1;
		}
		else if (percent[FaceShapeAnimations::FaceShapeAnimations_JawOpen] > 0.7){
			faceText += L":-O\n";
			wcout << faceText << endl;
			return 2;
		}
		else{
			faceText += L":-|\n";
			wcout << faceText << endl;
			return 3;
		}
		return 3;*/
		return testData;
}

void signInitialFaceValue(double initialVal[])
{

	for (int i = 0; i < FaceShapeAnimations_Count; i++)
	{
		FaceShapeAnimations faceAnim = (FaceShapeAnimations)i;
		{ switch (faceAnim)
			{
			case FaceShapeAnimations::FaceShapeAnimations_JawOpen:
				initialVal[i] = 0.05;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LeftcheekPuff:
				initialVal[i] = 0.03;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LefteyebrowLowerer:
				initialVal[i] = 0.2;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_RighteyebrowLowerer:
				initialVal[i] = 0.2;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LipCornerDepressorLeft:
				initialVal[i] = 0.38;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LipCornerDepressorRight:
				initialVal[i] = 0.5;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LipPucker:
				initialVal[i] = 0.3;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LipStretcherLeft:
				initialVal[i] = 0.03;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_LipStretcherRight:
				initialVal[i] = 0.03;
				break;
			case FaceShapeAnimations::FaceShapeAnimations_RightcheekPuff:
				initialVal[i] = 0.03;
				break;

			}
		}

	}
}

void readTrainingData(string fileName, struct svm_problem& problem){
	ifstream inFile;
	double data[TOTAL_DATA][3];
	problem.y = new double[FaceShapeAnimations_Count];
	problem.x = new svm_node*[TOTAL_DATA];
	problem.l = TOTAL_DATA;
	for (int i = 0; i < TOTAL_DATA; i++) problem.x[i] = new svm_node[4];

	inFile.open(fileName);
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < TOTAL_DATA; j++){
			inFile >> data[j][i];
			cout << data[j][i] << endl;
			problem.x[j][i].index = i;
			inFile >> problem.x[j][i].value;
		}
	}
	for (int i = 0; i < TOTAL_DATA; i++){
		problem.x[i][3].index = -1;
		if(i<210) problem.y[i] = 0;
		else if (i<420) problem.y[i] = 1;
		else if (i<630) problem.y[i] = 2;
		else problem.y[i] = 3;
	}
	inFile.close();
}

void setNewParameter(svm_parameter& parameter){
	parameter.svm_type = C_SVC;
	parameter.kernel_type = RBF;
	parameter.gamma = 0.3333;
	parameter.cache_size = 40;
	parameter.eps=0.1;
	parameter.C=1;
	parameter.shrinking = 1;
	parameter.probability = 0;
}

void getEigenVec(double eigenVec[][3]){
	ifstream inFile;
	inFile.open("eigenVec.txt");
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < FaceShapeAnimations_Count; j++){
			inFile >> eigenVec[j][i];
		}
	}
	inFile.close();
}

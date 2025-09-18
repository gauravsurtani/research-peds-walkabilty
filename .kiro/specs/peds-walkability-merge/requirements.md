# Requirements Document

## Introduction

This feature merges two existing PEDS (Pedestrian Environment Data Scan) walkability analysis systems into a unified, comprehensive platform. The current Streamlit application provides interactive Street View image analysis with SAM (Segment Anything Model) for computer vision, while the archived FastAPI application offers robust Places API integration with detailed walkability scoring algorithms. The merged system will combine the strengths of both applications to create a complete walkability analysis platform that leverages both visual analysis and geospatial data.

## Requirements

### Requirement 1

**User Story:** As a walkability researcher, I want to analyze street segments using both computer vision and geospatial data, so that I can get comprehensive walkability scores that consider both visual features and nearby amenities.

#### Acceptance Criteria

1. WHEN a user selects a location on the interactive map THEN the system SHALL fetch Street View imagery for that location
2. WHEN Street View imagery is available THEN the system SHALL run SAM automatic mask generation to identify visual features
3. WHEN a location is selected THEN the system SHALL query the Places API to find nearby amenities within a configurable radius
4. WHEN both visual and geospatial data are collected THEN the system SHALL calculate a comprehensive walkability score
5. IF Street View imagery is not available THEN the system SHALL still provide walkability analysis based on Places API data alone

### Requirement 2

**User Story:** As a user, I want an intuitive web interface to explore walkability data, so that I can easily select locations and view analysis results with visualizations.

#### Acceptance Criteria

1. WHEN the application loads THEN the system SHALL display an interactive map with default location markers
2. WHEN a user clicks on the map THEN the system SHALL update the selected coordinates and display them
3. WHEN analysis is complete THEN the system SHALL display Street View images with segmentation overlays
4. WHEN analysis is complete THEN the system SHALL show walkability scores with detailed breakdowns by category
5. WHEN masks are generated THEN the system SHALL provide downloadable binary mask files in ZIP format

### Requirement 3

**User Story:** As a researcher, I want detailed walkability scoring based on multiple factors, so that I can understand what contributes to pedestrian-friendly environments.

#### Acceptance Criteria

1. WHEN Places API data is collected THEN the system SHALL categorize places into walkability-relevant groups (transit, healthcare, retail, restaurants, amenities, services, entertainment)
2. WHEN place categories are determined THEN the system SHALL calculate density metrics per kilometer for each category
3. WHEN density metrics are calculated THEN the system SHALL apply weighted scoring algorithms with configurable weights
4. WHEN visual features are detected THEN the system SHALL incorporate computer vision results into the scoring model
5. WHEN scoring is complete THEN the system SHALL provide detailed score breakdowns showing contribution of each factor

### Requirement 4

**User Story:** As a developer, I want a modular architecture that supports both interactive analysis and API-based batch processing, so that the system can scale for different use cases.

#### Acceptance Criteria

1. WHEN the system is designed THEN it SHALL maintain separate modules for Places API client, SAM processing, and scoring engines
2. WHEN API endpoints are implemented THEN they SHALL support both single-location analysis and batch processing of multiple segments
3. WHEN the Streamlit interface is used THEN it SHALL leverage the same underlying analysis modules as the API endpoints
4. WHEN configuration changes are made THEN they SHALL be centralized and affect both UI and API components
5. IF the system encounters API rate limits THEN it SHALL implement appropriate backoff and retry mechanisms

### Requirement 5

**User Story:** As a system administrator, I want proper error handling and monitoring capabilities, so that I can ensure reliable operation and troubleshoot issues effectively.

#### Acceptance Criteria

1. WHEN Google Maps API keys are missing or invalid THEN the system SHALL provide clear error messages with remediation steps
2. WHEN Street View metadata indicates unavailable imagery THEN the system SHALL guide users on API configuration requirements
3. WHEN SAM model checkpoints are missing or incompatible THEN the system SHALL provide specific guidance on model selection
4. WHEN API rate limits are exceeded THEN the system SHALL log usage statistics and implement appropriate delays
5. WHEN errors occur THEN the system SHALL log detailed information for debugging while showing user-friendly messages

### Requirement 6

**User Story:** As a user, I want flexible configuration options for analysis parameters, so that I can customize the analysis for different research needs.

#### Acceptance Criteria

1. WHEN configuring Street View parameters THEN the system SHALL allow adjustment of image size, field of view, pitch, and heading
2. WHEN configuring Places API searches THEN the system SHALL allow customization of search radius and place types
3. WHEN configuring SAM processing THEN the system SHALL support different model types (vit_h, vit_l, vit_b) with automatic detection
4. WHEN configuring scoring weights THEN the system SHALL allow modification of category importance in walkability calculations with healthcare having highest priority (30% weight)
5. WHEN saving configurations THEN the system SHALL persist settings using environment variables and configuration files

### Requirement 7

**User Story:** As a researcher, I want comprehensive place categorization with healthcare prioritization, so that I can analyze walkability with focus on accessibility to medical facilities.

#### Acceptance Criteria

1. WHEN categorizing places THEN the system SHALL prioritize healthcare facilities (hospitals, doctors, dentists, pharmacies) with 30% scoring weight
2. WHEN categorizing places THEN the system SHALL include transit access (bus stations, subway, train stations) with 20% scoring weight
3. WHEN categorizing places THEN the system SHALL categorize amenities (banks, libraries, schools, parking) with 15% scoring weight
4. WHEN categorizing places THEN the system SHALL include retail, restaurants, and services with 10% weight each
5. WHEN categorizing places THEN the system SHALL include entertainment venues (theaters, museums, parks, gyms) with 5% scoring weight

### Requirement 8

**User Story:** As a system architect, I want a modular data acquisition and processing pipeline, so that the system can handle both real-time analysis and batch processing efficiently.

#### Acceptance Criteria

1. WHEN processing geographic coordinates THEN the system SHALL route data through separate acquisition modules for Street View and Maps APIs
2. WHEN Street View images are acquired THEN the system SHALL process them through a computer vision engine with SAM segmentation and object detection
3. WHEN API data is acquired THEN the system SHALL process it through structured data processing modules
4. WHEN features are extracted THEN the system SHALL construct unified feature matrices combining visual and API-derived features
5. WHEN feature matrices are complete THEN the system SHALL apply trained prediction models to generate PEDS walkability scores

### Requirement 9

**User Story:** As a data scientist, I want support for model training and validation workflows, so that I can improve walkability prediction accuracy using ground truth data.

#### Acceptance Criteria

1. WHEN ground truth PEDS audit data is available THEN the system SHALL support importing manual audit scores for training
2. WHEN training data is prepared THEN the system SHALL generate feature matrices from audited locations
3. WHEN supervised learning is performed THEN the system SHALL support training prediction models using ground truth scores
4. WHEN models are trained THEN the system SHALL provide validation metrics and model performance evaluation
5. WHEN models are validated THEN the system SHALL support deployment of trained models for inference on new locations

### Requirement 10

**User Story:** As a developer, I want proper caching and performance optimization, so that the system can handle repeated analyses efficiently while respecting API rate limits.

#### Acceptance Criteria

1. WHEN Street View metadata is fetched THEN the system SHALL cache results with configurable TTL to avoid repeated API calls
2. WHEN Street View images are downloaded THEN the system SHALL implement on-disk caching with TTL management
3. WHEN Places API calls are made THEN the system SHALL implement rate limiting with configurable requests per minute (default 100)
4. WHEN API quotas are approached THEN the system SHALL implement exponential backoff and usage monitoring
5. WHEN SAM models are loaded THEN the system SHALL cache model instances to avoid repeated loading overhead
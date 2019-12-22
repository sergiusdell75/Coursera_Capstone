# Coursera_Capstone
 This notebook will mainly be used for the capstone project

### Description of the problem and a discussion of the background

A relocation is stressful. Ideally, a new hire should be able to fully concentrate on the job requirements without spending lots of time to settle in. A real-estate agency is often approached to find a temporary accommodation for new employees.  Depending on the preferences (profile) of the new hire, the estate agent usually provides the best matches based on the experience (classical approach). However the experience only reflects the past without a projection into the future.

A modern approach based on the data analytics enables to allocate the best possible place for the temporary accommodation based on the following categories: venues,  rental (monthly) costs, crimes, schools (only applies if the new hire has got children). Moreover, the historical data can be used to make some predictions on the neighborhood of the current accommodation,e.g.  w.r.t. crime, to exclude or put a more weight on this particular  neighborhood for the future clients.

The business need: optimize the rent of the temporary accommodation. Better matches imply fully satisfied clients and a quicker turn around. Based on a profile of the client search for a best-fit place for the temporary accommodation. Moreover, predictions about  the  neighborhood could save time in decision making, which implies  a better performance.

In this project, I used Berlin as a place to move in.

### Description of the data and how it will be used to solve the problem

Based on the predefined categories (crimes, schools rental costs, venues) collect corresponding data sets (four in total).
Travel sites on the Internet, e.g., FourSquare www.foursquare.com, are ideally  to collect data on city venues.
Governments sites are suitable to collect data about historical crimes. Real estate agency sites are preferable for scrapping the average monthly rent. Local council sites are a good candidate to collect data on the schools, i.e., OFSTED reports.


Venue data were collected data as followed: 
1. Query the FourSqaure website for the top sites in Berlin;
2. Use the FourSquare API to get top restaurants and shopping recommendations closest to each of the top site  within a distance.

Crime data were collected as follows:
1. Use open source crime statistics for Berlin  https://www.statistik-berlin-brandenburg.de/regionalstatistiken/r-gesamt_neu.asp?Ptyp=410&Sageb=12015&creg=BBB&anzwer=6  
2. Download corresponding historical data as an excel file.

Rental costs data were collected as follows:
1. Use
2. Scrap for the average monthly rental costs.

School data were collected as follows:
1. Use https://www.gymnasium-berlin.net/abiturdaten/2018
2. Scrap for the avarage grade of schools in the borough for years 2012-2018


#### Data Preparation 

The purpose of this step is a transformation of the collected data into a useable subset.  It composes reading in different excel tables, scrubbing the webpages and saving data as pre-processed  in CSV-files. 

Crime data: the excel table was read in. The first two sheets were ignored. The sheets named  „Fallzahlen\*“  were transformed into panda data frames. The first four rows were dropped in every data frame.  The columns were renamed by translating into english.

School data: scrap for grades using  BeautifulSoup. Save data frames to csv files.

Rental costs data:

Venue data:


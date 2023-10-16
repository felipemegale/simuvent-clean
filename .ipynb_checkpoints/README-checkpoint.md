# Simuvent
Master's Thesis simulator

## The Data
### TomTom API
From TomTom's public API (free tier), I collected two kinds of data, totalling three different datasets. The first kind is called Flow, and the second kind is called Incidents. The Flow kind contains information collected by TomTom through deployed devices and apps running in drivers' mobile phones. The geographical locations chosen to collect data from were the Yonge St. & Front St. intersection, and Gardiner Expressway, both located in downtown Toronto, Canada, and each request to this API was made every 5 minutes. To create the first data set, a point in Gardiner Expressway was chosen, and TomTom's API provided me with data regarding a stretch of road of approximately 7.75km in length. Calls to this API endpoint returned, among other system-specific information, current travel speed, current travel time and road closure status. There are also free flow speed and free flow travel time, which are not used in this work due to the fact that what matters for this study is the actual speed that is recorded in the instant, not what the speed would be if there was no traffic. Since TomTom's APIs can be used as SDKs to build your own GPS app or website, the Flow data is focused on providing a broad picture of the current status of traffic in the location. To create the second data set, the data collected from the Yonge St. & Front St. intersection contains the same attributes, but the length of road is shorter, spanning approximately 280m and goes from the aforementionend collection point, northbound to Yonge St. & King St. The attributes for this second Flow location are the same as the attributes described for Gardiner Expressway Flow data. On the other hand, to make up the third data set collected from TomTom, the calls made to TomTom's Incidents endpoint requires two latitude-longitude pairs to make up a bounding box. Incidents reported inside this bounding box are returned when making a request to that endpoint. The attributes in this data set include what kind of incident is happening, the magnitude of the delay caused by such incident, when the incident started and when it should end, the length of road impacted by the incident, probability of occurrence, how many reports were made about that same incident, among others that were not relevant for the development of this work.

### Data Provided by the City of Toronto
In contact with the city of Toronto, I downloaded two additional data sets. Both datasets contain information about the volume of vehicles. Each entry in either data set shows how many vehicles were detected by inductino loops in a 15-minute time span.

## Setting up a simulation
- Download a map from OpenStreetMap (map.osm file)

- Convert `map.osm` to Sumo format: `netconvert --osm-files map.osm -o map.net.xml`
    - If you want to visualize/edit the generated map, you can open in netedit: `netedit map.net.xml`
    - Also, `netgenerate` can create a map for you

- Generate vehicle traffic using `randomTrips.py`
    - `randomTrips.py -n map.net.xml -r routes.rou.xml -o trips.xml --random-depart --random --verbose`
    - The above command will:
        - Generate a `routes.rou.xml` file and a `trips.xml` file
            - The `.rou.xml` file contains the routes each vehicle will take
        - Generate `3600` trips
        - Distribute departures randomly between begin and end (`--random-depart`)
        - Use a random seed do initialize RNG (`--random`)

- Get traces using `sumo -c <config_file>.sumocfg --fcd-output <file_name>.xml`
    - The `--device.fcd.period <time_step>` option can also be used to change the how often trace data will be saved

## A sample .sumocfg file
```xml
<configuration>
    <input>
        <net-file value=".net.xml"/>
        <route-files value="grid.rou.xml"/>
        <additional-files value="rerouter.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="10000"/>
    </time>
    <output>
        <fcd-output value="grid.output.xml"/>
    </output>
</configuration>
```
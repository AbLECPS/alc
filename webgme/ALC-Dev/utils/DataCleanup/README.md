## Data Cleanup
Data Cleanup is a utility for the ALC toolchain intended to clean up unused data directories.
This utility is very minimal, and some familiarity with the code is recommended before attempting to use it.
There is currently no input file or arguments, and configuring the utility requires editing the python script.
Additionally, the utility currently only offers the ability to MOVE data to an archive location for review by the user, it cannot DELETE the data.
This is intended to prevent accidental deletion, and requires the user to manually remove the data after running the utility.


### Background
The ALC Toolchain often generates large amounts of data which are not suitable for storing directly inside of the WebGME version-controlled database.
Instead, the data itself is stored on a more typical filesystem on an appropriate server (usually the same one running the ALC WebGME instance).
A metadata object describing the data (including a URI for locating the data) is returned to WebGME and stored in the database. 


Often, metadata objects will be deleted from the WebGME database when the associated data is no longer needed.
However, deleting the metadata object does not (currently) delete the actual data it identifies.
This commonly leads to large amounts of "orphaned" data - data which is no longer identified by any metadata object and therefore not reachable from the WebGME GUI.
Conversely, metadata objects may become "invalid" if the data it identifies no longer exists.
The primary function of the Data Cleanup utility is to discover all orphaned data and invalid metadata as well as provide options for cleaning up the orphaned data.


Data in ALC is organized based on the associated WebGME project and stored in a directory corresponding to the project which created the data.
For example, an experiment run by project *P1* may create a data artifact *D1* which will be stored in the data directory for project *P1*. 
A corresponding metadata object *M1* will be created in project *P1* such that *M1 -> D1*.
However, *M1* is not necessarily the only metadata object which points to data *D1*.
A second project *P2* may be duplicated from *P1* causing a separate metadata object *M2* to be created in *P2* such that *M2 -> D2*.
In this case, *M1* and *M2* both point to the same data *D1*.
If metadata *M1* is deleted, then *D1* is no longer associated with any metadata object in the parent project. 
However, this does NOT mean *D1* is orphaned since *M2* in project *P2* still points to *D1*.  


### Algorithm
- Start with a list of WebGME projects to be scanned/cleaned
- For every project build two objects:
    1. Index of all valid data objects owned/created by this project (data index)
    2. Index of all metadata objects which exist in this project (metadata index)
- After data index and metadata index has been built for every known project, iterate over all known project again to perform:
    - For every object in this project's metadata index, verify that it points to a valid object in the data index of ANY known project. If not, then this metadata is invalid.
        - **NOTE**: Due to the cross-project references discussed above, matadata objects do not necessarily point to a data artifact in the same project.
        - Each time a data object is pointed to by a valid metadata, also mark that data object as having at least one valid reference
- For each project, iterate through validated metadata index & data index. 
    - Provide user with a summary of valid/invalid metadata and valid/orphaned data.
    - Provide user options to list the invalid/orphaned objects, move all orphaned data to an archive location, or continue to the next project


### Known Issues
- The Data Cleanup only "knows" about the WebGME projects that were initially provided by the user. 
  It does NOT do any kind of automatic project discovery.
  Returning to the earlier example, it is possible a third project *P3* exists containing metadata *M3* such that *M3 -> D1*.
  However, if Data Cleanup is only aware of projects *P1, P2* and metadata objects *M1, M2* have been deleted, then the utility will report that *D1* has been orphaned.
  In reality, *D1* would still be a valid data object since *M3 -> D1*.
  Care is needed when specifying the list of projects to clean in order to avoid this problem.
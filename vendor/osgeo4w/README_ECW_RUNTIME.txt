Bundled ECW Runtime for Terra View

To make ECW files open on client computers without installing OSGeo4W locally, place a legal redistributable GDAL + ECW runtime here before building the EXE/installer.

Required minimum layout:

vendor/osgeo4w/bin/gdal_translate.exe
vendor/osgeo4w/bin/gdalinfo.exe
vendor/osgeo4w/bin/*.dll
vendor/osgeo4w/apps/gdal/share/gdal/   (if available)
vendor/osgeo4w/apps/gdal/lib/gdalplugins/ or vendor/osgeo4w/lib/gdalplugins/ containing ECW plugin
vendor/osgeo4w/share/proj/ or vendor/osgeo4w/apps/proj/share/proj/

Terra View checks this bundled path first, then C:\OSGeo4W, then QGIS, then PATH.

Important: ECW support depends on GDAL ECW/JP2 SDK licensing. Only bundle binaries that you have the right to redistribute.

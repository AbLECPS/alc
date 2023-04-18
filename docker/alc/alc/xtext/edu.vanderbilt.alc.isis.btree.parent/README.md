This is the top level for BTree DSL based on XText.
This project is used to setup the language server with vscode.

It assumes that 
  -`JAVA_HOME` is set and points to a `JAVA 11` runtime.
  -`code` (visual studio code application) is in the path. 

Build and install the extension
  - Launch a terminal, cd to this folder and type the following command

```
./gradlew clean installDist installExtension
```

Once the above step is complete, if you want to start a VS code with the example/ demo  folder type

```
./gradlew startCode
```
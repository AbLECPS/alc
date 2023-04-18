 define(['js/logger',
 'js/Dialogs/CodeEditor/CodeEditorDialog',
 'js/Controls/PropertyGrid/Widgets/AssetWidget',
 'jquery-fancytree/jquery.fancytree',
 'jquery-fancytree/jquery.fancytree.table',
 'jquery-fancytree/jquery.fancytree.gridnav',
 'css!./styles/ActivityDefinitionTree.css'
], function (Logger, CodeEditorDialog, AssetWidget) {
 'use strict';

    function ActivityChoicesDefinitionTree(container, options) {

        this._logger = Logger.create('gme:Widgets:ActivityDefinition:ActivityChoicesDefinitionTree',
            WebGMEGlobal.gmeConfig.client.log);

        options = options || {};

        this._client = WebGMEGlobal.Client;

        this._el = container;

        this._initialize(options);

        this._logger.debug('Ctor finished...');
    }

    ActivityChoicesDefinitionTree.prototype._initialize = function (options) {
        var self = this;
        this._el.html('');

        let table = '<table id="choices-tree"> \
            <colgroup> \
            <col width="5%" /> \
            <col width="50%"/> \
            <col width="20%" /> \
            <col width="15%" /> \
            <col width="10%" /> \
            </colgroup> \
            <thead> \
            <tr> \
                <th>#</th> \
                <th>Parameter</th> \
                <th>Value</th> \
                <th>Type</th> \
                <th>isDefault</th> \
            </tr> \
            </thead> \
        </table>';

        this._treeEl = $(table, {});

        this._treeEl.fancytree({
            titlesTabbable: true,
            extensions: ["table", "gridnav"],
            table: {
              indentation: 20,
              nodeColumnIdx: 1
            },
            gridnav: {
              autofocusInput: false,
              handleCursorKeys: true,
            },
            click: function(event, data) {

            },
            dblclick: function(event, data)
            {

            },
            lazyLoad: function (event, data) {
                self._logger.debug('onLazyRead node:' + data.node.key);
                self.onNodeOpen.call(self, data.node.key);
                event.preventDefault();
                event.stopPropagation();
            },
            createNode: function(event, data) {
              var node = data.node,
                $tdList = $(node.tr).find(">td");

              $tdList.eq(2).css("display", "flex");

              $tdList.eq(2).css("align-items", "center");

              $tdList.eq(2).css("justify-content", "center");

              $tdList.eq(0).css("background-color","white");

              $tdList.eq(2).css("width","100%");

              $tdList.eq(2).css("padding","4px");

              if (node.isFolder()) {
                $tdList
                  .eq(1)
                  .prop("colspan", 5)
                  .nextAll()
                  .remove();
              }
              else
              {
                $tdList.eq(4).text("false");

                if(node.data.type === 'integer')
                {
                  self.initInput($tdList, data, "number", self);
                }
                else if(node.data.type === 'float' ||
                        node.data.type === 'string' ||
                        node.data.type.length < 1)
                {
                  self.initInput($tdList, data, "text", self);
                }
                else if(node.data.type === 'boolean')
                {
                  self.initInput($tdList, data, "checkbox", self);
                }
                else if(node.data.type === 'array' ||
                        node.data.type === 'code' ||
                        node.data.type === 'dictionary')
                {
                  self.initInput($tdList, data, "button", self);
                }
                else if(node.data.type === 'asset' ||
                        node.data.type === 'LEC' ||
                        node.data.type === 'Data')
                {
                  self.initAsset($tdList, data, self);
                }
                
                $tdList.eq(3).text(node.data.type);
              }
            },
            renderColumns: function(event, data) {
              var node = data.node,
                $tdList = $(node.tr).find(">td");

              $tdList.eq(0).text(node.getIndexHier());
            },
            modifyChild: function(event, data) {
                data.tree.info(event.type, data);
            },
          })
          .on("nodeCommand", function(event, data) {
            var refNode,
              moveMode,
              tree = $.ui.fancytree.getTree(this),
              node = tree.getActiveNode();

            switch (data.cmd) {
              case "addChild":
              case "addSibling":
              case "indent":
              case "moveDown":
              case "moveUp":
              case "outdent":
              case "remove":
              case "rename":
              case "cut":
              case "copy":
              case "clear":
              case "paste":
                break;
              default:
                alert("Unhandled command: " + data.cmd);
                return;
            }
        });

        this._treeInstance = this._treeEl.fancytree('getTree');
        this._el.append(this._treeEl);
    };

    ActivityChoicesDefinitionTree.prototype.initInput = function (tdList, data, inputType, self)
    {
      tdList.eq(2).append($("<input id=" + data.node.data.name + " type=" + inputType + " />", {}));
      
      $("#"+data.node.data.name)[0].value = data.node.data.value;
      if(data.node.data.value.length < 1)
      {
        $("#"+data.node.data.name)[0].value = data.node.data.default;
        tdList.eq(4).text("true");
      }

      if(inputType === 'button' ||
         data.node.data.type === 'string')
      {
        $("#"+data.node.data.name)[0].style = "width:100%;text-align:center";
      }
      else if(inputType === 'checkbox')
      {
        var checked = ($("#"+data.node.data.name)[0].value === 'True') ? true : false;
        $("#"+data.node.data.name)[0].checked = checked;

        $("#"+data.node.data.name)[0].style = "width:100%;text-align:right";
      }
      else
      {
        $("#"+data.node.data.name)[0].style = "width:100%;text-align:right";
      }

      if(data.node.data.value === data.node.data.default)
      {
        tdList.eq(4).text("true");
      }

      if(inputType === "button")
      {
        $("#"+data.node.data.name).on("click", { source: data.node.data, node: data.node, this: self }, self.handleInput);
        if($("#"+data.node.data.name)[0].value.length > 15)
        {
          $("#"+data.node.data.name)[0].value = $("#"+data.node.data.name)[0].value.slice(0,15);
        }
      }
      else
      {
        $("#"+data.node.data.name).on("input", { source: data.node.data, node: data.node, this: self }, self.handleInput);
      }
    };

    ActivityChoicesDefinitionTree.prototype.initAsset = function (tdList, data, self)
    {
      let val = data.node.data.value;
      if(val.length < 1)
      {
        val = data.node.data.default;
      }
      
      var dialog = new AssetWidget({
          name: 'asset-manager-widget',
          id: data.node.data.id,
          value: val
      });
      dialog.onFinishChange((code_data) => {
          if (!code_data.newValue) {
              self._logger.debug('No file uploaded.');
              return;
          }
          if(code_data.newValue === data.node.data.default)
          {
            tdList.eq(4).text("true");
          }
          else
          {
            tdList.eq(4).text("false");
          }
          if(code_data.newValue !== data.node.data.value)
          {
            tdList.eq(0).css("background-color","#6495ED");
          }
          else
          {
            tdList.eq(0).css("background-color","white");
          }
          
          self._client.setAttribute(data.node.data.id, "asset", code_data.newValue);
      });
      tdList.eq(2).append(dialog.el[0]);
      $(".widget")[0].style = "width:150px";
      $(".asset-widget")[0].style = "width:150px";
      if(data.node.data.value === data.node.data.default)
      {
        tdList.eq(4).text("true");
      }
    };

    ActivityChoicesDefinitionTree.prototype.arrayChange = function (event, tdList) 
    {
      var self = event.data.this;
      var choice = new CodeEditorDialog();
      var val = event.data.source.value;
      if(val.length === 0)
      {
        val = event.data.source.default;
      }

      var params ={
          "name": "value",
          "value": val,
          "multilineType": "plaintext",
          "activeObject": event.data.source.id,
          "activeSelection": [event.data.source.id],
          "title": "Array",
          "readOnly":false
      };

      choice.show(params);

      choice._saveBtn[0].addEventListener("click", function(evt){
        self.checkArrayEditor(choice, event, tdList);
      });

      choice._okBtn[0].addEventListener("click", function(evt){
        self.checkArrayEditor(choice, event, tdList);
      });
    };

    ActivityChoicesDefinitionTree.prototype.dictionaryChange = function (event, tdList) 
    {
      var self = event.data.this;
      var choice = new CodeEditorDialog();
      var val = event.data.source.value;
      if(val.length === 0)
      {
        val = event.data.source.default;
      }

      var params ={
          "name": "value",
          "value": val,
          "multilineType": "plaintext",
          "activeObject": event.data.source.id,
          "activeSelection": [event.data.source.id],
          "title": "Dictionary",
          "readOnly":false
      };

      choice.show(params);

      choice._saveBtn[0].addEventListener("click", function(evt){
        self.checkDictionaryEditor(choice, event, tdList);
      });

      choice._okBtn[0].addEventListener("click", function(evt){
        self.checkDictionaryEditor(choice, event, tdList);
      });
    };

    ActivityChoicesDefinitionTree.prototype.checkDictionaryEditor = function (choice, event, tdList) 
    {
      var self = event.data.this;
      let txt = choice._savedValue;
      let val = event.data.source.value;
      if(val.length < 1)
      {
        val = event.data.source.default;
      }
      if(txt !== val)
      {
        tdList.eq(0).css("background-color","#6495ED");
      }
      else
      {
        tdList.eq(0).css("background-color","white");
      }
      if(txt === event.data.source.default)
      {
        tdList.eq(4).text("true");
      }
      txt = txt.replace(/\s+/g,'');
      let checkArr = new RegExp(/^\{.*[\}]$/,'g');
      if(checkArr.test(txt))
      {
        $("#" + event.data.source.name)[0].style = 'color:black';
        self._client.setAttribute(event.data.source.id, "value", choice._savedValue);
      }
      else
      {
        $("#" + event.data.source.name)[0].style = 'color:red';
      }

      let sval = choice._savedValue;
      if(choice._savedValue.length > 10)
      {
        sval = choice._savedValue.slice(0,10);
      }
      $("#"+event.data.source.name)[0].value = sval;
    };

    ActivityChoicesDefinitionTree.prototype.floatChange = function (event, tdList) 
    {
      var self = event.data.this;
      let txt = event.currentTarget.value;
      let val = event.data.source.value;
      if(val.length < 1)
      {
        val = event.data.source.default;
      }
      if(txt !== val)
      {
        tdList.eq(0).css("background-color","#6495ED");
      }
      else
      {
        tdList.eq(0).css("background-color","white");
      }

      if(txt === event.data.source.default)
      {
        tdList.eq(4).text("true");
      }

      txt = txt.replace(/\s+/g,'');
      txt = txt + "\n";
      let checkScientific = new RegExp(/(^\.\d+\e[\-\+]\d+)|(^0\.\d*\e[\-\+]\d+)|((^1\.\d*\e\-\d+)|(^1\.0*\e\+\d+))/,'g');
      let checkDecimal = new RegExp(/^(\.\d+|0\.\d*|(1\.\n|1\.0+\n))/,'g');
      if(checkScientific.test(txt) ||
          checkDecimal.test(txt))
      {
        $("#" + event.data.source.name)[0].style = 'color:black';
        self._client.setAttribute(event.data.source.id, "value", event.currentTarget.value);
      }
      else
      {
        $("#" + event.data.source.name)[0].style = 'color:red';
      }
    };

    ActivityChoicesDefinitionTree.prototype.integerChange = function (event, tdList) 
    {
      var self = event.data.this;
      let txt = event.currentTarget.value;
      let val = event.data.source.value;
      if(val.length < 1)
      {
        val = event.data.source.default;
      }
      if(txt !== val)
      {
        tdList.eq(0).css("background-color","#6495ED");
      }
      else
      {
        tdList.eq(0).css("background-color","white");
      }

      if(txt === event.data.source.default)
      {
        tdList.eq(4).text("true");
      }
      txt = txt.replace(/\s+/g,'');
      txt = txt + "\n";
      let checkInt = new RegExp(/^\d+\n/,'g');
      if(checkInt.test(txt))
      {
        $("#" + event.data.source.name)[0].style = 'color:black';
        self._client.setAttribute(event.data.source.id, "value", event.currentTarget.value);
      }
      else
      {
        $("#" + event.data.source.name)[0].style = 'color:red';
      }
    };

    ActivityChoicesDefinitionTree.prototype.checkArrayEditor = function (choice, event, tdList)
    {
        var self = event.data.this;
        let txt = choice._savedValue;
        let val = event.data.source.value;
        if(val.length < 1)
        {
          val = event.data.source.default;
        }
        if(txt !== val)
        {
          tdList.eq(0).css("background-color","#6495ED");
        }
        else
        {
          tdList.eq(0).css("background-color","white");
        }
        if(txt === event.data.source.default)
        {
          tdList.eq(4).text("true");
        }
        txt = txt.replace(/\s+/g,'');
        let checkArr = new RegExp(/(^\[(\"[A-Za-z\/\-\_0-9]*\"\,)*\n(\"[A-Za-z\/\-\_0-9]*\"\,)*\"[A-Za-z\/\-\_0-9]*\"\])|(^\[(\"[A-Za-z\/\-\_0-9]*\"\,)*(\"[A-Za-z\/\-\_0-9]*\"\]))|(^\[((\d+,)+\d+|(\d*\.\d+,)+(\d*\.\d+))\]|^\[((\d+,)+\d+|(\d*\.\d+,)+(\d*\.\d+))\,\n((\d+,)+\d+|(\d*\.\d+,)+(\d*\.\d+))\])/,'g');
        if(checkArr.test(txt))
        {
          $("#" + event.data.source.name)[0].style = 'color:black';
          self._client.setAttribute(event.data.source.id, "value", choice._savedValue);
        }
        else
        {
          $("#" + event.data.source.name)[0].style = 'color:red';
        }

        let sval = choice._savedValue;
        if(choice._savedValue.length > 10)
        {
          sval = choice._savedValue.slice(0,10);
        }
        $("#"+event.data.source.name)[0].value = sval;
    };

    ActivityChoicesDefinitionTree.prototype.codeChange = function (event, tdList)
    {
      var self = event.data.this;
      var choice = new CodeEditorDialog();
      var val = event.data.source.value;
      if(val.length === 0)
      {
        val = event.data.source.default;
      }

      var params ={
          "name": "value",
          "value": val,
          "multilineType": event.data.source.codeType,
          "activeObject": event.data.source.id,
          "activeSelection": [event.data.source.id],
          "title": "Code",
          "readOnly":false
      };

      choice.show(params);

      choice._saveBtn[0].addEventListener("click", function(evt){
        self.checkCodeEditor(choice, event, tdList);
      });

      choice._okBtn[0].addEventListener("click", function(evt){
        self.checkCodeEditor(choice, event, tdList);
      });
    };

    ActivityChoicesDefinitionTree.prototype.checkCodeEditor = function (choice, event, tdList)
    {
      var self = event.data.this;
      let sval = choice._savedValue;
      let val = event.data.source.value;
      if(val.length < 1)
      {
        val = event.data.source.default;
      }
      if(sval !== val)
      {
        tdList.eq(0).css("background-color","#6495ED");
      }
      else
      {
        tdList.eq(0).css("background-color","white");
      }
      if(choice._savedValue.length > 10)
      {
        sval = choice._savedValue.slice(0,10);
      }
      $("#"+event.data.source.name)[0].value = sval;

      if(sval === event.data.source.default)
      {
        tdList.eq(4).text("true");
      }
      self._client.setAttribute(event.data.source.id, "value", choice._savedValue);
    };

    ActivityChoicesDefinitionTree.prototype.booleanChange = function (event, tdList)
    {
      var self = event.data.this;
      let txt = event.currentTarget.checked;
      let val = event.data.source.value;
      if(val.length < 1)
      {
        val = event.data.source.default;
      }
      if(typeof(val) === "string")
      {
        val  = (val.toLocaleLowerCase() === 'true') ? true : false;
      }
      if(txt !== val)
      {
        tdList.eq(0).css("background-color","#6495ED");
      }
      else
      {
        tdList.eq(0).css("background-color","white");
      }

      var defval  = (event.data.source.default.toLocaleLowerCase() === 'true') ? true : false;
      if(txt === defval)
      {
        tdList.eq(4).text("true");
      }
      else
      {
        tdList.eq(4).text("false");
      }
      self._client.setAttribute(event.data.source.id, "value", "False");
      if(txt)
      {
        self._client.setAttribute(event.data.source.id, "value", "True");
      }
    };

    ActivityChoicesDefinitionTree.prototype.stringChange = function (event, tdList)
    {
      var self = event.data.this;
      let txt = event.currentTarget.value;
      let val = event.data.source.value;
      if(val.length < 1)
      {
        val = event.data.source.default;
      }
      if(txt !== val)
      {
        tdList.eq(0).css("background-color","#6495ED");
      }
      else
      {
        tdList.eq(0).css("background-color","white");
      }
      if(txt === event.data.source.default)
      {
        tdList.eq(4).text("true");
      }
      self._client.setAttribute(event.data.source.id, "value", txt);
    };

    ActivityChoicesDefinitionTree.prototype.handleInput = function (event) {
        var self = event.data.this;
        var $tdList = $(event.data.node.tr).find(">td");
        $tdList.eq(4).text("false");
        if(event.data.source.type === 'array')
        {
          self.arrayChange(event, $tdList);
        }
        else if(event.data.source.type === 'float')
        {
          self.floatChange(event, $tdList);  
        }
        else if(event.data.source.type === 'integer')
        {
          self.integerChange(event, $tdList);
        }
        else if(event.data.source.type === 'code')
        {
          self.codeChange(event, $tdList);
        }
        else if(event.data.source.type === 'boolean')
        {
          self.booleanChange(event, $tdList);
        }
        else if(event.data.source.type === 'string')
        {
          self.stringChange(event, $tdList);
        }
        else if(event.data.source.type === 'dictionary')
        {
          self.dictionaryChange(event, $tdList);
        }
    };

    ActivityChoicesDefinitionTree.prototype.createNode = function (parentNode, objDescriptor) {
        var newNode;
        objDescriptor.name = objDescriptor.name || '';

        if (parentNode === null) 
        {
            parentNode = this._treeInstance.getRootNode();
        }

        newNode = parentNode.addChildren({
            title: objDescriptor.name,
            tooltip: objDescriptor.data ? objDescriptor.data.description : '',
            key: objDescriptor.id,
            folder: objDescriptor.folder,
            lazy: objDescriptor.hasChildren,
            extraClasses: objDescriptor.class || '',
            icon: objDescriptor.icon || null,
            isConnection: objDescriptor.isConnection,
            isAbstract: objDescriptor.isAbstract,
            isLibrary: objDescriptor.isLibrary,
            isLibraryRoot: objDescriptor.isLibraryRoot,
            libraryInfo: objDescriptor.libraryInfo,
            metaType: objDescriptor.metaType,
            isMetaNode: objDescriptor.isMetaNode, 
            data: objDescriptor.data
        });

        this._logger.debug('New node created.');

        return newNode;
    };

    ActivityChoicesDefinitionTree.prototype.onNodeOpen = function (nodeId) {
        this._logger.warn('Default onNodeOpen for node ' +
            nodeId + ' called, doing nothing. Please override onNodeOpen(nodeId)');
    };

 return ActivityChoicesDefinitionTree;
});
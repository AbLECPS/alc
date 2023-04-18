define([
    'js/util',
    'text!./GSNIDGENUserChoiceDialog.html',
	'css!./GSNIDGenUserChoiceDialog.css'], 
	function (util, IDGENUserChoiceDialogTemplate) {
    'use strict';
    var IDGENUserChoiceDialog;
    IDGENUserChoiceDialog = function () {
    };

    IDGENUserChoiceDialog.prototype.show = function (saveCallBack) {
        var self = this;

        this._initDialog(saveCallBack);
		
		this._dialog.modal('show');
		
		this._dialog.on('shown.bs.modal', function () {
        });


        this._dialog.on('hidden.bs.modal', function () {
            self._dialog.remove();
            self._dialog.empty();
            self._dialog = undefined;
        });
		
		
    };

    IDGENUserChoiceDialog.prototype._initDialog = function (saveCallBack) {
        var self = this,
            closeSave;

        closeSave = function () {
            self._dialog.modal('hide');

            if (saveCallBack) {
				var i;
				var value=0;
				for(i=0; i<self._radio1.length; i+=1)
				{
					if (self._radio1[i].checked)
					{
						value=self._radio1[i].value;
						break;
					}
				}
				//alert(value);
                saveCallBack.call(self, value);
            }
        };

        this._dialog = $(IDGENUserChoiceDialogTemplate);

        //get controls
        this._el = this._dialog.find('.mo-body').first();

        this._btnSave = this._dialog.find('.btn-save').first();

       this._radio1 = this._dialog.find(':radio');
	   
	   if (this._radio1)
	   {
			self._radio1[0].checked =1;
	   }
	   
	   this.checked=1;
	   
	   $("input:radio[name='rGroup']").click(function() {
			self.checked= $("input:radio[name='rGroup']:checked").val();
		});
       

        //click on SAVE button
        this._btnSave.on('click', function (event) {
            closeSave();

            event.stopPropagation();
            event.preventDefault();
        });
        
    };


    return IDGENUserChoiceDialog;
});
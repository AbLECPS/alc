/*globals define, $*/
define([
    'deepforge/viz/Utils',
    'text!./ModelRow.html'
], function(
    Utils,
    ROW_HTML
) {
    'use strict';
    
    var ModelItem = function(parent, node) {
        this.$el = $(ROW_HTML);
        this.initialize();
        this.time = this.update(node);
        //parent.append(this.$el);
    };

    ModelItem.prototype.initialize = function() {
        // Get the fields and stuff
        this.$name = this.$el.find('.name');
        this.$type = this.$el.find('.type');
        this.$size = this.$el.find('.result');
        this.$log  = this.$el.find('.log');
        this.$createdAt = this.$el.find('.createdAt');
        this.$download = this.$el.find('.data-download');
        this.$delete = this.$el.find('.data-remove');
        this.colors = {'Finished':"lightgreen","Finished_w_Errors":"#ff9999","Submitted":"lightyellow","Started":"lightblue"}
        
    };

    ModelItem.prototype.update = function(node) {
        var date = node.createdAt ? Utils.getDisplayTime(node.createdAt) : 'unknown';
        

        this.$name.text(node.name);
        this.$type.text(node.type || 'unknown');
        var mtext  = '-'
        var status = node.jobstatus;
		    if (node.resultURL)
        {
            this.$size.empty();
            this.$size.append('<a  href="'+node.resultURL+'" target="_blank"> Result </a>')
            if (!status || status == "Unknown")
            {
                status = 'Finished'
            }
        }
        else {
            this.$size.empty();
            this.$size.text('-')
        }
        mtext  = '-'
		if (node.logURL)
        {
            if (status && ((status == "Started" )))
            {
              this.$log.empty();
              this.$log.append('<a  href="'+node.logURL+'" target="_blank"> Log </a>  <br>  <a  href="'+node.viewURL+'" target="_blank"> Sim </a>')
            }
            else{
                this.$log.empty();
                this.$log.append('<a  href="'+node.logURL+'" target="_blank"> Log </a>')
            }
        }
        else {
            this.$log.empty();
            this.$log.text('-')
        }
        
        this.$download.attr('href', node.dataURL);
        this.$createdAt.text(date);
        if (status && status != "Unknown" )
        {
            this.$size.parent().css("background-color", this.colors[status]);
            
        }
        

        

        return date;
    };

    ModelItem.prototype.remove = function() {
        this.$el.remove();
    };

    return ModelItem;
});

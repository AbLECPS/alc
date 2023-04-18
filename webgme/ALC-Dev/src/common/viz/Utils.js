/* globals define*/
define({
    getDisplayTime: timestamp => {
       var options = { day: '2-digit',month: '2-digit',year: 'numeric', hour: '2-digit',  minute: '2-digit', second: '2-digit'};
    
        var today = new Date().toLocaleDateString('en-US',options),
            date = new Date(timestamp).toLocaleDateString('en-US',options),
            time = new Date(timestamp).toLocaleTimeString('en-US',options);
            date = `${date}`;// (${time})`;
            return date;

        /*if (date === today) {
            date = `Today (${time})`;
        }
        else {
            date = `${date} (${time})`;
        }
        return date;*/
    },
    ClassForJobStatus: {
        success: 'success',
        canceled: 'job-canceled',
        failed: 'danger',
        pending: '',
        running: 'warning'
    }
});

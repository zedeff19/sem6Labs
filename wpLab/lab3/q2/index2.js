function updateTime() {
    const currentTime = new Date();
    const hours = String(currentTime.getHours()).padStart(2, '0');
    const minutes = String(currentTime.getMinutes()).padStart(2, '0');
    const seconds = String(currentTime.getSeconds()).padStart(2, '0');
    
    const timeString = `${hours}:${minutes}:${seconds}`;
    
    // Update the DOM element with the current time
    document.getElementById('current-time').textContent = timeString;

    // console.log(typeof(minutes));
    if (minutes == '00')
    {
        // console.log("its been an hour")
        
        let greeting = '';
    if (hours >= 5 && hours < 12) {
        greeting = "Good Morning";
    } else if (hours >= 12 && hours < 17) {
        greeting = "Good Afternoon";
    } else {
        greeting = "Good Evening";
    }

    document.getElementById('greeting').textContent = `${greeting} Tanay, it's ${hours}:${minutes}`;    

    
    }
    
}

// Update time every second (1000ms)
setInterval(updateTime, 1000);

// Initially update the time when the page loads
updateTime();

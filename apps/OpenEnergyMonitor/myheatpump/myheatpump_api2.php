<?php

function get_daily_stats($feed,$app,$start,$end,$starting_power) {

    $timezone = 'Europe/London';
    $date = new DateTime();
    $date->setTimezone(new DateTimeZone($timezone));

    if ($end===null || $start===null) {
        $date->modify("midnight");
        $end = $date->getTimestamp();
        $date->modify("-30 day");
        $start = $date->getTimestamp();
    } else {
        $start = convert_time($start,$timezone);
        $end = convert_time($end,$timezone);
        
        $date->setTimestamp($start);
        $date->modify("midnight");
        $start = $date->getTimestamp();
    }
    
    $out = "";
    $fields = array();
    $fields[] = "timestamp";

    $categories = ["combined","running","space","water"];

    foreach ($categories as $category) {
        $fields[] = $category."_elec_kwh";
        $fields[] = $category."_heat_kwh";
        $fields[] = $category."_cop";
        $fields[] = $category."_data_length";
        $fields[] = $category."_elec_mean";
        $fields[] = $category."_heat_mean";
        $fields[] = $category."_flowT_mean";
        $fields[] = $category."_returnT_mean";
        $fields[] = $category."_outsideT_mean";
        $fields[] = $category."_roomT_mean";
        $fields[] = $category."_prc_carnot";
    }
    
    $fields[] = "combined_cooling_kwh";
    $fields[] = "from_energy_feeds_elec_kwh";
    $fields[] = "from_energy_Feeds_heat_kwh";
    $fields[] = "from_energy_Feeds_cop";
    
    $fields[] = "quality_elec";
    $fields[] = "quality_heat";
    $fields[] = "quality_flowT";
    $fields[] = "quality_returnT";
    $fields[] = "quality_outsideT";
    $fields[] = "quality_roomT";

    $out .= implode(",",$fields)."\n";
        
    $time = $start;
    while ($time<$end) {
        // print $date->format("c")."\n";
        
        $stats = get_heatpump_stats($feed,$app,$time,$time+(3600*24),$starting_power);
        
        $values = array();

        $values[] = $stats['start'];

        foreach ($categories as $category) {
            $values[] = $stats['stats'][$category]['elec_kwh'];
            $values[] = $stats['stats'][$category]['heat_kwh'];
            $values[] = $stats['stats'][$category]['cop'];
            $values[] = $stats['stats'][$category]['data_length'];
            $values[] = $stats['stats'][$category]['elec_mean'];
            $values[] = $stats['stats'][$category]['heat_mean'];
            $values[] = $stats['stats'][$category]['flowT_mean'];
            $values[] = $stats['stats'][$category]['returnT_mean'];
            $values[] = $stats['stats'][$category]['outsideT_mean'];
            $values[] = $stats['stats'][$category]['roomT_mean'];
            $values[] = $stats['stats'][$category]['prc_carnot'];
        }
        
        $values[] = $stats['stats']["combined"]['cooling_kwh'];
        $values[] = $stats['stats']['from_energy_feeds']['elec_kwh'];
        $values[] = $stats['stats']['from_energy_feeds']['heat_kwh'];
        $values[] = $stats['stats']['from_energy_feeds']['cop'];
           
        $values[] = $stats['quality']['elec'];
        $values[] = $stats['quality']['heat'];
        $values[] = $stats['quality']['flowT'];
        $values[] = $stats['quality']['returnT'];
        $values[] = $stats['quality']['outsideT'];
        $values[] = $stats['quality']['roomT'];
                
        $out .= implode(",",$values)."\n";
        
        $date->modify("+1 day");
        $time = $date->getTimestamp();
    }

    return $out;
}

function get_heatpump_stats($feed,$app,$start,$end,$starting_power) {

    // --------------------------------------------------------------------------------------------------------------    
    // Validate params
    // --------------------------------------------------------------------------------------------------------------
    if ($end===null || $start===null) {
        $date = new DateTime();
        $date->setTimezone(new DateTimeZone("Europe/London"));
        $date->modify("midnight");
        $end = $date->getTimestamp();
        $date->modify("-30 day");
        $start = $date->getTimestamp();
    } else {
        $timezone = 'Europe/London';
        $start = convert_time($start,$timezone);
        $end = convert_time($end,$timezone);
    }
    
    if ($end<=$start) return array('success'=>false, 'message'=>"Request end time before start time");

    $period = $end-$start;
    if ($period<=3600*24*7) {
        $interval = 10;
    } else if ($period<=3600*24*14) {
        $interval = 20;
    } else if ($period<=3600*24*21) {
        $interval = 30;
    } else if ($period<=3600*24*42) {
        $interval = 60;
    } else if ($period<=3600*24*90) {
        $interval = 120;
    } else {
        return array('success'=>false, 'message'=>"period to large");
    }
    
    //$interval = 60;
    
    
    if (!isset($app->config->heatpump_elec) || $app->config->heatpump_elec<1) return array('success'=>false, 'message'=>"Missing electricity consumption feed");
        
    // --------------------------------------------------------------------------------------------------------------    
    // Load data
    // --------------------------------------------------------------------------------------------------------------    
    $data = array();
    
    $elec_meta = $feed->get_meta($app->config->heatpump_elec);
    
    $feeds = array("heatpump_elec","heatpump_flowT","heatpump_returnT","heatpump_outsideT","heatpump_roomT","heatpump_heat","heatpump_dhw");
    
    foreach ($feeds as $key) {
        $data[$key] = false;
        if (isset($app->config->$key) && $app->config->$key>0) {   
            $data[$key] = $feed->get_data($app->config->$key,$start,$end-$interval,$interval,1,"Europe/London","notime");
            $data[$key] = remove_null_values($data[$key],$interval);
        }
    }
    
    $cop_stats = calculate_window_cops($data, $interval, $starting_power);
    
    $stats = process_stats($data, $interval, $starting_power);
    
    foreach ($stats as $category => $val) {
        $cop_stats[$category]["elec_mean"] = $stats[$category]["elec"]["mean"];    
        $cop_stats[$category]["heat_mean"] = $stats[$category]["heat"]["mean"];    
        $cop_stats[$category]["flowT_mean"] = $stats[$category]["flowT"]["mean"];
        $cop_stats[$category]["returnT_mean"] = $stats[$category]["returnT"]["mean"];
        $cop_stats[$category]["outsideT_mean"] = $stats[$category]["outsideT"]["mean"];
        $cop_stats[$category]["roomT_mean"] = $stats[$category]["roomT"]["mean"]; 
    }
    
    $ideal_carnot_heat_mean = carnot_simulator($data, $starting_power);
    foreach ($stats as $category => $val) {
        $cop_stats[$category]["prc_carnot"] = null;
        if ($ideal_carnot_heat_mean[$category]!==null && $ideal_carnot_heat_mean[$category]>0) {
            $cop_stats[$category]["prc_carnot"] = number_format(100 * $stats[$category]["heat"]["mean"] / $ideal_carnot_heat_mean[$category],3,'.','')*1;
        }
    }
    
    $cop_stats["combined"]["cooling_kwh"] = process_cooling($data,$interval);
    
    $elec_kwh = get_cumulative_kwh($feed,$app->config->heatpump_elec_kwh,$start,$end);
    $heat_kwh = get_cumulative_kwh($feed,$app->config->heatpump_heat_kwh,$start,$end);
    
    $cop = null;
    if ($elec_kwh>0) {
        $cop = $heat_kwh / $elec_kwh;
    }
    if ($elec_kwh!==null) $elec_kwh = number_format($elec_kwh,4,'.','')*1;
    if ($heat_kwh!==null) $heat_kwh = number_format($heat_kwh,4,'.','')*1;
    if ($cop!==null) $cop = number_format($cop,3,'.','')*1;
    
    $cop_stats["from_energy_feeds"] = array(
        "elec_kwh" => $elec_kwh,
        "heat_kwh" => $heat_kwh,
        "cop" => $cop
    );
    
    $result = [
      "start"=>(int)$start,
      "end"=>(int)$end,
      "interval"=>(int)$interval,
      "stats"=>$cop_stats,
      //"stats"=>$stats,
      "quality"=>[
        "elec"=>get_quality($data["heatpump_elec"]),
        "heat"=>get_quality($data["heatpump_heat"]),
        "flowT"=>get_quality($data["heatpump_flowT"]),
        "returnT"=>get_quality($data["heatpump_returnT"]),
        "outsideT"=>get_quality($data["heatpump_outsideT"]),
        "roomT"=>get_quality($data["heatpump_roomT"])
      ]
    ];
    
    return $result;
}

function process_stats($data, $interval, $starting_power) {
    $stats = [
        'combined' => [],
        'running' => [],
        'space' => [],
        'water' => []
    ];
    
    $feed_options = [
        "elec" => ["name" => "Electric consumption", "unit" => "W", "dp" => 1],
        "heat" => ["name" => "Heat output", "unit" => "W", "dp" => 1],
        //"heat_carnot" => ["name" => "Simulated heat output", "unit" => "W", "dp" => 0],
        "flowT" => ["name" => "Flow temperature", "unit" => "°C", "dp" => 2],
        "returnT" => ["name" => "Return temperature", "unit" => "°C", "dp" => 2],
        "outsideT" => ["name" => "Outside temperature", "unit" => "°C", "dp" => 2],
        "roomT" => ["name" => "Room temperature", "unit" => "°C", "dp" => 2],
        //"targetT" => ["name" => "Target temperature", "unit" => "°C", "dp" => 1],
        //"flowrate" => ["name" => "Flow rate", "unit" => "", "dp" => 3]
    ];

    foreach ($feed_options as $key => $props) {
        foreach ($stats as $x => $val) {
            $stats[$x][$key] = [
                'sum' => 0,
                'count' => 0,
                'mean' => null,
                'kwh' => null,
                //'minval' => null,
                //'maxval' => null
            ];
        }
    }

    $dhw_enable = false;
    if (isset($data["heatpump_dhw"]) && $data["heatpump_dhw"] != false) {
        $dhw_enable = true;
    }

    for ($z = 0; $z < count($data["heatpump_elec"]); $z++) {
        $power = $data["heatpump_elec"][$z];

        $dhw = false;
        if ($dhw_enable) {
            $dhw = $data["heatpump_dhw"][$z];
        }

        foreach ($feed_options as $key => $props) {
            if (isset($data["heatpump_".$key][$z])) {
                $value = $data["heatpump_".$key][$z];
                if ($value !== null) {

                    $stats['combined'][$key]['sum'] += $value;
                    $stats['combined'][$key]['count']++;
                    //stats_min_max($stats, 'combined', $key, $value);

                    if ($power !== null && $power >= $starting_power) {
                        $stats['running'][$key]['sum'] += $value;
                        $stats['running'][$key]['count']++;
                        //stats_min_max($stats, 'running', $key, $value);

                        if ($dhw_enable) {
                            if ($dhw) {
                                $stats['water'][$key]['sum'] += $value;
                                $stats['water'][$key]['count']++;
                                //stats_min_max($stats, 'water', $key, $value);
                            } else {
                                $stats['space'][$key]['sum'] += $value;
                                $stats['space'][$key]['count']++;
                                //stats_min_max($stats, 'space', $key, $value);
                            }
                        }
                    }
                }
            }
        }
    }

    foreach ($stats as $x => $val) {
        foreach ($feed_options as $key => $props) {

            $stats[$x][$key]["mean"] = null;
            if ($stats[$x][$key]["count"] > 0) {
                $stats[$x][$key]["mean"] = $stats[$x][$key]["sum"] / $stats[$x][$key]["count"];
            }
            /*
            $stats[$x][$key]["diff"] = null;
            if ($stats[$x][$key]["minval"] !== null && $stats[$x][$key]["maxval"] !== null) {
                $stats[$x][$key]["diff"] = $stats[$x][$key]["maxval"] - $stats[$x][$key]["minval"];
            }
            */
        }
    }
    
    foreach ($stats as $x => $val) {
        foreach ($feed_options as $key => $props) {
        
            if ($props["unit"] == "W" && $stats[$x][$key]["mean"] !== null) {
                $stats[$x][$key]["kwh"] = ($stats[$x][$key]["mean"] * $stats[$x][$key]["count"] * $interval) / 3600000;
                $stats[$x][$key]["kwh"] = number_format($stats[$x][$key]["kwh"],4,".","")*1;            
            } else {
                unset($stats[$x][$key]["kwh"]);
            }
        
            //if ($stats[$x][$key]["sum"]!==null) {
            //    $stats[$x][$key]["sum"] = number_format($stats[$x][$key]["sum"],$props["dp"],".","")*1;
            //}
            unset($stats[$x][$key]["sum"]);
            unset($stats[$x][$key]["count"]);
            
            if ($stats[$x][$key]["mean"]!==null) {
                $stats[$x][$key]["mean"] = number_format($stats[$x][$key]["mean"],$props["dp"],".","")*1;
            }
            /*
            if ($stats[$x][$key]["minval"]!==null) {
                $stats[$x][$key]["minval"] = number_format($stats[$x][$key]["minval"],$props["dp"],".","")*1;  
            }
            if ($stats[$x][$key]["maxval"]!==null) {
                $stats[$x][$key]["maxval"] = number_format($stats[$x][$key]["maxval"],$props["dp"],".","")*1;  
            }    
            if ($stats[$x][$key]["diff"]!==null) {
                $stats[$x][$key]["diff"] = number_format($stats[$x][$key]["diff"],$props["dp"],".","")*1;  
            }*/
             
        }
    }

    return $stats;
}

function stats_min_max(&$stats, $category, $key, $value) {
    // This function should update the min and max values in $stats array
    // Initialize min and max if they are null
    if ($stats[$category][$key]['minval'] === null || $value < $stats[$category][$key]['minval']) {
        $stats[$category][$key]['minval'] = $value;
    }
    if ($stats[$category][$key]['maxval'] === null || $value > $stats[$category][$key]['maxval']) {
        $stats[$category][$key]['maxval'] = $value;
    }
}


// Remove null values from feed data
function remove_null_values($data, $interval) {
    $last_valid_pos = 0;
    for ($pos = 0; $pos < count($data); $pos++) {
        if ($data[$pos] !== null) {
            $null_time = ($pos - $last_valid_pos) * $interval;
            if ($null_time < 900) {
                for ($x = $last_valid_pos + 1; $x < $pos; $x++) {
                    $data[$x] = $data[$last_valid_pos];
                }
            }
            $last_valid_pos = $pos;
        }
    }
    return $data;
}

function get_quality($data) {
    $count = count($data);
    if ($count<1) return 0;
    
    $null_count = 0;
    for ($pos = 0; $pos < $count; $pos++) {
        if (is_null($data[$pos])) {
            $null_count ++;
        }
    }
    $quality = 100*(1-($null_count / $count));
    return number_format($quality,3,'.','')*1;
}

function calculate_window_cops($data, $interval, $starting_power) {
    $cop_stats = array(
        "combined" => array(),
        "running" => array(),
        "space" => array(),
        "water" => array(),
    );

    foreach ($cop_stats as $category => $value) {
        $cop_stats[$category]["elec_kwh"] = 0;
        $cop_stats[$category]["heat_kwh"] = 0;
        $cop_stats[$category]["data_length"] = 0;
    }

    if (isset($data["heatpump_elec"]) && isset($data["heatpump_heat"])) {

        $dhw_enable = false;
        if (isset($data["heatpump_dhw"]) && $data["heatpump_dhw"] != false) {
            $dhw_enable = true;
        }

        $power_to_kwh = 1.0 * $interval / 3600000.0;

        foreach ($data["heatpump_elec"] as $z => $elec_data) {
            $elec = $data["heatpump_elec"][$z];
            $heat = $data["heatpump_heat"][$z];

            $dhw = false;
            if ($dhw_enable) {
                $dhw = $data["heatpump_dhw"][$z];
            }

            if ($elec !== null && $heat !== null) {
                $cop_stats["combined"]["elec_kwh"] += $elec * $power_to_kwh;
                $cop_stats["combined"]["heat_kwh"] += $heat * $power_to_kwh;
                $cop_stats["combined"]["data_length"] += $interval;
                
                if ($elec >= $starting_power) {
                    $cop_stats["running"]["elec_kwh"] += $elec * $power_to_kwh;
                    $cop_stats["running"]["heat_kwh"] += $heat * $power_to_kwh;
                    $cop_stats["running"]["data_length"] += $interval;

                    if ($dhw_enable) {
                        if ($dhw) {
                            $cop_stats["water"]["elec_kwh"] += $elec * $power_to_kwh;
                            $cop_stats["water"]["heat_kwh"] += $heat * $power_to_kwh;
                            $cop_stats["water"]["data_length"] += $interval;
                        } else {
                            $cop_stats["space"]["elec_kwh"] += $elec * $power_to_kwh;
                            $cop_stats["space"]["heat_kwh"] += $heat * $power_to_kwh;
                            $cop_stats["space"]["data_length"] += $interval;
                        }
                    }
                }
            }
        }

        foreach ($cop_stats as $category => $stats) {
        
            $cop_stats[$category]["cop"] = 0;
            if ($cop_stats[$category]["elec_kwh"] > 0) {
                $cop_stats[$category]["cop"] = $cop_stats[$category]["heat_kwh"] / $cop_stats[$category]["elec_kwh"];
            }

            $cop_stats[$category]["elec_kwh"] = number_format($cop_stats[$category]["elec_kwh"],4,".","")*1;            
            $cop_stats[$category]["heat_kwh"] = number_format($cop_stats[$category]["heat_kwh"],4,".","")*1;            
            $cop_stats[$category]["cop"] = number_format($cop_stats[$category]["cop"],3,".","")*1;
            
            if ($cop_stats[$category]["data_length"] == 0) {
                $cop_stats[$category]["elec_kwh"] = null;
                $cop_stats[$category]["heat_kwh"] = null;
                $cop_stats[$category]["cop"] = null;
            }
        }
    }
    return $cop_stats;
}

function carnot_simulator($data, $starting_power) {
    if (!isset($data["heatpump_elec"])) return false;
    if (!isset($data["heatpump_flowT"])) return false;
    if (!isset($data["heatpump_returnT"])) return false;
    if (!isset($data["heatpump_outsideT"])) return false;
    
    $dhw_enable = false;
    if (isset($data["heatpump_dhw"]) && $data["heatpump_dhw"] != false) {
        $dhw_enable = true;
    }

    $condensing_offset = 2;
    $evaporator_offset = -6;
    
    $combined_ideal_carnot_heat_sum = 0;
    $combined_carnot_heat_n = 0;
    
    $running_ideal_carnot_heat_sum = 0;
    $running_carnot_heat_n = 0;
    
    $space_ideal_carnot_heat_sum = 0;
    $space_carnot_heat_n = 0;
    
    $water_ideal_carnot_heat_sum = 0;
    $water_carnot_heat_n = 0;  

    foreach ($data["heatpump_elec"] as $z => $value) {
        $elec = $data["heatpump_elec"][$z];
        $flowT = $data["heatpump_flowT"][$z];
        $returnT = $data["heatpump_returnT"][$z];
        $ambientT = $data["heatpump_outsideT"][$z];
        
        $dhw = false;
        if ($dhw_enable) {
            $dhw = $data["heatpump_dhw"][$z];
        }

        $a = $flowT + $condensing_offset + 273;
        $b = $a - ($ambientT + $evaporator_offset + 273);
        
        $carnot_COP = 0;
        if ($b!=0) {
            $carnot_COP = $a / $b;
        }
        
        if ($elec !== null && $carnot_COP !== null) {
        
            $ideal_carnot_heat = $elec * $carnot_COP;
            if ($returnT > $flowT) {
                $ideal_carnot_heat *= -1;
            }
        
            $combined_ideal_carnot_heat_sum += $ideal_carnot_heat;
            $combined_carnot_heat_n++;
        
            if ($elec >= $starting_power) {
                $running_ideal_carnot_heat_sum += $ideal_carnot_heat;
                $running_carnot_heat_n++;
                
                if ($dhw_enable) {
                    if ($dhw) {
                        $water_ideal_carnot_heat_sum += $ideal_carnot_heat;
                        $water_carnot_heat_n++;
                    } else {
                        $space_ideal_carnot_heat_sum += $ideal_carnot_heat;
                        $space_carnot_heat_n++;    
                    }
                }
            }
        }
    }
    
    $combined_ideal_carnot_heat_mean = null;
    if ($combined_carnot_heat_n>0) {
        $combined_ideal_carnot_heat_mean = round($combined_ideal_carnot_heat_sum / $combined_carnot_heat_n);
    }
    
    $running_ideal_carnot_heat_mean = null;
    if ($running_carnot_heat_n>0) {
        $running_ideal_carnot_heat_mean = round($running_ideal_carnot_heat_sum / $running_carnot_heat_n);
    }
    
    $space_ideal_carnot_heat_mean = null;
    if ($space_carnot_heat_n>0) {
        $space_ideal_carnot_heat_mean = round($space_ideal_carnot_heat_sum / $space_carnot_heat_n);
    }
    
    $water_ideal_carnot_heat_mean = null;
    if ($water_carnot_heat_n>0) {
        $water_ideal_carnot_heat_mean = round($water_ideal_carnot_heat_sum / $water_carnot_heat_n);
    }

    return array(
        "combined" => $combined_ideal_carnot_heat_mean,
        "running" => $running_ideal_carnot_heat_mean,
        "space" => $space_ideal_carnot_heat_mean,
        "water" => $water_ideal_carnot_heat_mean
    );
}

function process_cooling($data, $interval) {

    $power_to_kwh = 1.0 * $interval / 3600000.0;
    
    $total_negative_heat_kwh = 0;
    if (isset($data["heatpump_heat"])) {
        foreach ($data["heatpump_heat"] as $z => $value) {
            $heat = $data["heatpump_heat"][$z];
            if ($heat !== null && $heat < 0) {
                $total_negative_heat_kwh += -1 * $heat * $power_to_kwh;
            }
        }

    }
    return number_format($total_negative_heat_kwh,3,'.','')*1;
}

function get_cumulative_kwh($feed,$feedid,$start,$end) {
    
    $meta = $feed->get_meta($feedid);
    if ($meta->start_time>$start) {
        $start = $meta->start_time;
    }
    if ($meta->end_time<$end) {
        $end = $meta->end_time;
    }
    
    if ($end<$start) return false;
    
    
    $kwh_start = $feed->get_value($feedid,$start);
    $kwh_end = $feed->get_value($feedid,$end);
    //print $feedid." ".$start." ".$end." ".$kwh_start." ".$kwh_end." ";
    return $kwh_end - $kwh_start;
}

function convert_time($time,$timezone) {
    // Option to specify times as date strings
    if (!is_numeric($time)) {
        $date = new DateTime();
        $date->setTimezone(new DateTimeZone($timezone));
        $date->modify($time);
        $date->modify('midnight');
        $time = $date->getTimestamp();
    }    
    
    // If timestamp is in milliseconds convert to seconds
    if (($time/1000000000)>100) {
        $time *= 0.001;
    }
    return $time;
}
